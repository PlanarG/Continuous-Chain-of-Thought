import logging
import os
import sys
import json

def get_logger(args):
    os.makedirs(args.log_dir, exist_ok=True)
    logger          = logging.getLogger('CCoT')
    file_path       = os.path.join(args.log_dir, f"{args.timestamp}.log")
    file_handler    = logging.FileHandler(file_path)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter       = logging.Formatter(f'[%(asctime)s] [%(levelname)s] [{args.rank}] %(message)s')
    
    logger.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.debug(json.dumps(vars(args), indent=2))
    return logger