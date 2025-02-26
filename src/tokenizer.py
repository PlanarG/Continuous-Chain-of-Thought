import torch
from typing import List

token_dict = None

def construct_token_dict(
    num_range: int,
    special_tokens: List[str] = ["<pad>", "<bos>", "<eos>"], 
    operators: List[str] = ["+", "-", "*", "/", "(", ")", "="]
): 
    tokens = special_tokens + operators + [str(i) for i in range(num_range)]
    return {token: i for i, token in enumerate(tokens)}

def tokenizer(
    sentences: List[str],
    num_range: int,
):
    global token_dict
    if token_dict is None: 
        token_dict = construct_token_dict(num_range)
    
    token_ids = []
    for sentence in sentences:
        token_ids.append([token_dict[token] for token in ("<bos> " + sentence + " <eos>").split(' ')])

    maxlen = max([len(ids) for ids in token_ids]) + 1
    bz = len(token_ids)

    x = torch.zeros((bz, maxlen), dtype=torch.long)
    for i, ids in enumerate(token_ids):
        x[i, :len(ids)] = torch.tensor(ids)

    return x

if __name__ == "__main__":
    print(tokenizer(["1 + 11 = 12", "( 2 + 2 ) * 3 = 12"], 13))