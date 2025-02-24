from transformers import AutoModelForCausalLM
from src.model import CCoTConfig, CCoT
from src.args_parser import parse_arguments

args = parse_arguments()

config = CCoTConfig.load(args.model_config)
ccot = CCoT(config)
ccot.load_from_pretrained_standard_cot("Qwen/Qwen2.5-1.5B-Instruct")

# for name, param in ccot.named_parameters():
#     if param.requires_grad:
#         print(name, param.shape)