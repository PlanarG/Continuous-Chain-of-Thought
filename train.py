import torch
import wandb

from src.args_parser import parse_arguments
from src.logger import get_logger
from src.model import CCOT
from src.tokenizer import tokenizer
from dataset.dataset import Dataset

from accelerate import Accelerator

args    = parse_arguments()
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
args.rank = accelerator.process_index

logger  = get_logger(args)
dataset = Dataset(args, logger)
device  = accelerator.device
model   = CCOT(args, logger).to(device)

train_dataset = torch.utils.data.DataLoader(
    dataset.train["dataset"], 
    batch_size=args.per_device_train_batch_size,
    shuffle=True
)

optimizer = torch.optim.AdamW(model.parameters())
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataset)

total_steps = (len(train_dataloader) * args.num_train_epochs) // args.gradient_accumulation_steps
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=total_steps, pct_start=args.warmup_ratio)

for epoch in range(args.num_train_epochs):
    for batch in train_dataloader:
        with accelerator.accumulate(model):
            outputs = model(batch["input"])
            loss = compute_loss(outputs, batch["output"])
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            scheduler.step()
            