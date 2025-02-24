import torch

from src.args_parser import parse_arguments
from src.logger import get_logger
from src.model import CCoT
from dataset.dataset import Dataset

from accelerate import Accelerator

accelerator = Accelerator()

args    = parse_arguments(accelerator.process_index)
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
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataset)