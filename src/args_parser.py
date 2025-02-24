import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="CCoT")

    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save data")
    parser.add_argument("--model_config", type=str, default="config/Qwen-1.5B-config.json", help="Model configuration file")

    args = parser.parse_args()
    return args