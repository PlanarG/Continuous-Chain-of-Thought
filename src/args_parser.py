import argparse
import yaml
import time
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Path to YAML config file")
    parser.add_argument('--train', action="store_true", help="Train the model", default=False)
    parser.add_argument('--eval', action="store_true", help="Evaluate the model", default=False)
    args = parser.parse_args()

    if not args.train and not args.eval:
        raise ValueError("Either --train or --eval should be set")

    if args.train and args.eval:
        raise ValueError("Only one of --train or --eval should be set")

    configs = ["config/defaults.yml"]
    if args.train:
        configs.append("config/train/train-defaults.yml")
    if args.eval:
        configs.append("config/eval/eval-defaults.yml")
    if args.config:
        configs.append(args.config)
    
    for config in configs:
        if not os.path.exists(config):
            raise FileNotFoundError(f"Config file {config} not found")

        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            for key, value in config.items():
                setattr(args, key, value)

    timestamp       = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    args.timestamp  = timestamp

    return args