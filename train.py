import torch
import wandb
import yaml
import random
import os

from tqdm import tqdm

from src.args_parser import parse_arguments
from src.logger import get_logger
from src.model import get_model
from src.tokenizer import tokenizer, decode
from dataset.dataset import Dataset

from accelerate import Accelerator

args = parse_arguments()
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
args.rank = accelerator.process_index

logger  = get_logger(args)
dataset = Dataset(args, logger)
device  = accelerator.device
model   = get_model(args, logger, device)
unwrapped_model = model

train_dataset = torch.utils.data.DataLoader(
    dataset.train["dataset"], 
    batch_size=args.per_device_train_batch_size,
    shuffle=True
)

eval_dataset = torch.utils.data.DataLoader(
    dataset.test["dataset"], 
    batch_size=args.per_device_eval_batch_size,
    shuffle=True
)

optimizer = torch.optim.AdamW(model.parameters())

total_steps = (len(train_dataset) * args.num_epochs) // args.gradient_accumulation_steps
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=args.warmup_ratio)

model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataset, eval_dataset, scheduler)

ckpt_save_path = os.path.join(args.ckpt_dir, args.timestamp)
if args.rank == 0:
    os.makedirs(ckpt_save_path, exist_ok=True)

    with open(os.path.join(ckpt_save_path, "config.yml"), "w") as f:
        yaml.dump(vars(args), f)

        logger.info(f"Config file saved to {ckpt_save_path}/config.yml")

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

global_step = 0

for epoch in range(args.num_epochs):

    pbar = train_dataloader
    if args.rank == 0:
        pbar = tqdm(pbar, desc=f"Epoch {epoch + 1}/{args.num_epochs}", total=len(train_dataloader))    

    for batch in pbar:
        global_step += 1

        input_ids = tokenizer(batch["input"], num_range=args.num_range).to(device)
        output_ids = tokenizer(batch["output"], num_range=args.num_range).to(device)
        label_ids = torch.zeros_like(output_ids).to(device)
        label_ids[:, :-1] = output_ids[:, 1:]

        num_loops = random.randint(5, 15)

        with accelerator.accumulate(model):
            logits = model(
                input_ids  = input_ids, 
                answer_ids = output_ids, 
                num_loops  = num_loops, 
                num_contemplation_tokens = 10
            )

            output_len = output_ids.shape[1]
            output_logits = logits[:, -output_len:, :]
            loss = criterion(output_logits.transpose(1, 2), label_ids)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


        if args.rank == 0 and global_step % args.logging_steps == 0:
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs} | Step {global_step} | Loss {loss.item()}")
            # wandb.log({"loss": loss.item()})
        
    if args.rank == 0 and epoch % args.save_epochs == 0:
        unwrapped_model = accelerator.unwrap_model(model)

        logger.info(f"Saving model checkpoint")
        torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_save_path, f"model-{epoch}.pth"))   
        logger.info(f"Model checkpoint saved to {ckpt_save_path}/model-{epoch}.pth")
    
    if (epoch + 1) % 5 == 0:
        model_unwrapped = accelerator.unwrap_model(model)

        max_eval_steps = len(eval_dataloader)
        if hasattr(args, "max_eval_steps"):
            max_eval_steps = min(max_eval_steps, args.max_eval_steps)

        pbar = eval_dataloader
        if args.rank == 0:
            pbar = tqdm(pbar, desc=f"Evaluating", total=max_eval_steps)

        acc_list = []
        eval_steps = 0
        for batch in pbar:
            input_ids = tokenizer(batch["input"], num_range=args.num_range).to(device)
            answers = batch["answer"].to(device)

            bz = input_ids.shape[0]

            with torch.no_grad():
                outputs, results = model_unwrapped.generate(input_ids, num_loops=10, num_contemplation_tokens=10)
                results_str = decode(results, to_list=True)
                corr = [results_str[i] == str(answers[i].item()) for i in range(bz)]
                acc = torch.tensor(corr, dtype=torch.float).to(device).mean()
            
            acc = accelerator.gather(acc)
            acc_list.append(acc.mean().item())
            eval_steps += 1

            if eval_steps >= max_eval_steps:
                break

        if args.rank == 0:
            logger.info(f"Accuracy: {sum(acc_list) / len(acc_list)}")

