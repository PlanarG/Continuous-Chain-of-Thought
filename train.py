import torch
import wandb
import yaml
import random
import os

from tqdm import tqdm

from src.args_parser import parse_arguments
from src.logger import get_logger
from src.model import CCoT
from src.tokenizer import tokenizer
from dataset.dataset import Dataset

from accelerate import Accelerator

args = parse_arguments()
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
args.rank = accelerator.process_index

logger  = get_logger(args)
dataset = Dataset(args, logger)
device  = accelerator.device
model   = CCoT(args, logger).to(device)

train_dataset = torch.utils.data.DataLoader(
    dataset.train["dataset"], 
    batch_size=args.per_device_train_batch_size,
    shuffle=True
)

optimizer = torch.optim.AdamW(model.parameters())

total_steps = (len(train_dataset) * args.num_epochs) // args.gradient_accumulation_steps
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=args.warmup_ratio)

model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataset, scheduler)

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
            loss = criterion(output_logits.transpose(1, 2), output_ids)

            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


        if args.rank == 0 and global_step % args.logging_steps == 0:
            logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs} | Step {global_step} | Loss {loss.item()}")
            # wandb.log({"loss": loss.item()})
        
    if args.rank == 0 and global_step % args.save_steps == 0:
        unwrapped_model = accelerator.unwrap_model(model)

        logger.info(f"Saving model checkpoint")
        torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_save_path, f"model-{global_step}.pth"))   
        logger.info(f"Model checkpoint saved to {ckpt_save_path}/model-{global_step}.pth")
    
