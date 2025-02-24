import os
import json
import numpy as np
import argparse
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

operators = ["+", "-", "*", "/"]
inv = None

# reduce two numbers with an operator
def reduce(a: int, b: int, op: str, num_range: int) -> int:
    match op:
        case "+": return (a + b) % num_range
        case "-": return (a - b + num_range) % num_range
        case "*": return (a * b) % num_range
        case "/": 
            if b == 0:
                raise ValueError("Division by zero")
            
            global inv
            if inv is None:
                inv = { i : j for i in range(num_range) for j in range(num_range) if i * j % num_range == 1 }
            
            return (a * inv[b]) % num_range
        
        case _: raise ValueError(f"Invalid operator: {op}")

# generate a random arithmetic expression with a solution trajectory
# the expression will not be surrounded by parentheses
# eg. get_expression(2, 10) -> [["1", "+", "2"], ["3"]]
# eg. get_expression(3, 10) -> [["(", "1", "+", "2", ")", "*", "3"], ["3", "*", "3"], ["9"]]
# length: the number of elements in the expression
# num_range: the range of the numbers. It should be a prime number.
def get_expression(length: int, num_range: int) -> List[List[str]]:
    if length == 1:
        return [[str(random.randint(0, 9))]]
    
    operator = random.choice(operators)
    left_length = random.randint(1, length - 1)
    right_length = length - left_length
    left = get_expression(left_length, num_range)

    while True:
        try: 
            right = get_expression(right_length, num_range)
            result = reduce(int(left[-1][0]), int(right[-1][0]), operator, num_range)

            # To get a more realistic synthetic dataset, the solution trajectory will be sampled randomly instead of being deterministic, eg. choose the left-most operator to reduce first.
            reduce_order = [1] * (left_length - 1) + [0] * (right_length - 1)
            random.shuffle(reduce_order)

            def merge_intermediate_step(left_cur, right_cur):
                left_expr_reduced, right_expr_reduced = left[left_cur], right[right_cur]
                if left_cur + 1 < left_length:
                    left_expr_reduced = ['('] + left_expr_reduced + [')'] 
                if right_cur + 1 < right_length:
                    right_expr_reduced = ['('] + right_expr_reduced + [')']

                return left_expr_reduced + [operator] + right_expr_reduced
            
            solution = [merge_intermediate_step(0, 0)]
            for i, left_cur in enumerate([sum(reduce_order[:i + 1]) for i in range(len(reduce_order))]):
                solution.append(merge_intermediate_step(left_cur, i + 1 - left_cur))
            
            solution.append([str(result)])
            break

        except ValueError:
            continue
    return solution


def convert(expression: List[List[str]]):
    return {
        "input"  : ' '.join(expression[0]) + ' =', 
        "output" : ' = '.join([' '.join(expr) for expr in expression[1:]]),
        "answer" : int(expression[-1][0])
    }

def get_batch_expression(args, split: str, num_samples: int, process_id: int):
    data, lengths = [], []

    bar = range(num_samples)
    if process_id == 0:
        bar = tqdm(bar, desc=f"Generating {split} dataset", total=num_samples)
    for _ in bar:
        length = args.length
        if not args.exact:
            # longer expressions are more likely to be sampled
            length = random.choices(range(2, length), weights=[i for i in range(2, length)])[0]

        entry = get_expression(length, args.num_range)
        data.append(convert(entry))
        lengths.append(sum(len(expr) for expr in entry))

    return data, lengths

class Dataset:
    def __init__(self, args, split: str, num_samples: int):
        self.args = args
        self.split = split
        self.num_samples = num_samples
        self.data = []

        data_lengths = []
        batches = np.array_split(range(num_samples), args.num_workers)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(
                get_batch_expression, 
                args, 
                split, 
                len(batch), 
                i
            ) for i, batch in enumerate(batches)]

            for future in futures:
                data, lengths = future.result()
                self.data.extend(data)
                data_lengths.extend(lengths)

        self.avg_length = sum(data_lengths) / len(data_lengths)
        self.max_length = max(data_lengths)

    def save(self):
        if self.num_samples == 0:
            return
        
        os.makedirs(os.path.join(self.args.data_dir, self.split), exist_ok=True)
        with open(os.path.join(self.args.data_dir, self.split, "data.json"), "w") as f:
            json.dump(self.data, f, indent=2)
        
        with open(os.path.join(self.args.data_dir, self.split, "info.json"), "w") as f:
            json.dump({
                "num_samples": self.num_samples,
                "avg_length":  self.avg_length, 
                "max_length":  self.max_length
            }, f, indent=2)

# python dataset/generate.py --length 10 --num_range 59 --train_samples 1000000 --test_samples 10000 --num_workers 4 --exact --data_dir dataset/full
# structure of the dataset:
# - dataset (specified by --data_dir)
#   - train
#     - data.json
#        [{"input": "1 + ( 2 * 3 ) =", "output": "1 + 6 = 7", "answer": 7}, ...]
#     - info.json
#   - test
#     - data.json
#     - info.json
#   - args.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length",        type=int,  default=10,      help="The number of elements in the expression")
    parser.add_argument("--num_range",     type=int,  default=59,      help="The range of the numbers. It should be a prime number.")
    parser.add_argument("--train_samples", type=int,  default=1000000, help="Size of the training dataset")
    parser.add_argument("--test_samples",  type=int,  default=10000,   help="Size of the test dataset")
    parser.add_argument("--num_workers",   type=int,  default=4,       help="Number of workers for multiprocessing")
    
    parser.add_argument("--exact",         action="store_true", default=False, help="The number of elements in the expression")
    parser.add_argument("--data_dir",      type=str,  default="dataset/full", help="Path to the dataset")

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    train = Dataset(args, "train", args.train_samples)
    test  = Dataset(args, "test",  args.test_samples)

    train.save()
    test.save()

    with open(os.path.join(args.data_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


    
    