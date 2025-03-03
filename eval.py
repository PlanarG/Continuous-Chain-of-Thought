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
accelerator = Accelerator()
args.rank = accelerator.process_index

logger  = get_logger(args)
dataset = Dataset(args, logger)
device  = accelerator.device
model   = get_model(args, logger, device)

model_unwrapped = model

test_dataset = torch.utils.data.DataLoader(
    dataset.test["dataset"], 
    batch_size=args.per_device_eval_batch_size,
    shuffle=True
)

model, test_dataloader = accelerator.prepare(model, test_dataset)

pbar = test_dataloader
if args.rank == 0:
    pbar = tqdm(pbar, desc="Evaluating", total=len(test_dataloader))

acc_list = []
for batch in pbar:
    input_ids = tokenizer(batch["input"], num_range=args.num_range).to(device)
    answers = batch["answer"].to(device)

    bz = input_ids.shape[0]

    with torch.no_grad():
        outputs, results = model_unwrapped.generate(input_ids, num_loops=20, num_contemplation_tokens=20, max_length=500)
        results_str = decode(results, to_list=True)
        corr = [results_str[i] == str(answers[i].item()) for i in range(bz)]
        acc = torch.tensor(corr, dtype=torch.float).to(device).mean()
    
    acc = accelerator.gather(acc)
    logger.info(f"Accuracy: {acc.mean().item()}")
    acc_list.append(acc.mean().item())

if args.rank == 0:
    logger.info(f"Accuracy: {sum(acc_list) / len(acc_list)}")
