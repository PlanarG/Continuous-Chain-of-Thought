import torch
import wandb
import yaml
import random
import os

from tqdm import tqdm

from src.args_parser import parse_arguments
from src.logger import get_logger
from src.model import CCoT
from src.tokenizer import tokenizer, decode
from dataset.dataset import Dataset

from accelerate import Accelerator

args = parse_arguments()
accelerator = Accelerator()
args.rank = accelerator.process_index

print(args)

logger  = get_logger(args)
dataset = Dataset(args, logger)
device  = accelerator.device
model   = CCoT(args, logger, device)

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
    output_ids = tokenizer(batch["output"], num_range=args.num_range).to(device)
    answers = batch["answer"].to(device)

    bz = input_ids.shape[0]

    with torch.no_grad():
        outputs, results = model_unwrapped.generate(input_ids, num_loops=10, num_contemplation_tokens=10)
        results_str = decode(results, to_list=True)
        corr = [results_str[i] == str(answers[i].item()) for i in range(bz)]
        acc = torch.tensor(corr, dtype=torch.float).to(device).mean()
    
    acc = accelerator.gather(acc)
    acc_list.append(acc.mean().item())

if args.rank == 0:
    logger.info(f"Accuracy: {sum(acc_list) / len(acc_list)}")
