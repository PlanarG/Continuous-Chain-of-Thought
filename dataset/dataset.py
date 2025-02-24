import os
import json
from datasets import load_dataset

class Dataset:
    def __init__(self, args, logger):
        self.args = args 
        self.logger = logger
        self.train, self.test = {}, {}

        self.load("train")
        self.load("test")
    
    def load(self, split: str):
        data = {}

        self.logger.info(f"Loading split {split} from dataset {self.args.data_dir}...")

        if os.path.exists(os.path.join(self.args.data_dir, split)):
            data["dataset"] = load_dataset(os.path.join(self.args.data_dir, split), data_files="data.json")["train"]
            with open(os.path.join(self.args.data_dir, split, "info.json"), "r") as f:
                data.update(json.load(f))

            num_samples = data["num_samples"]
            self.logger.info(f"{self.args.data_dir}/{split} is loaded with {num_samples} samples.")

            if split == "train":
                self.train = data
            else:
                self.test = data
        else:
            self.logger.warn(f"Split {split} does not exist.")
