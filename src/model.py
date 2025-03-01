from src.ccot import CCoT
from src.cot import CoT

model_class = {
    "CoT": CoT,
    "CCoT": CCoT,
}

def get_model(args, logger, device):
    model = model_class[args.model](args, logger, device)
    return model


