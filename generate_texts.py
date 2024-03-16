from typing import List, Tuple
from transformers import AutoTokenizer, LlamaModel, Trainer, DataCollatorWithPadding, LlamaForCausalLM
from datasets import load_from_disk
from functools import partial
from collections import namedtuple
from tqdm import trange

from transformers.modeling_outputs import BaseModelOutputWithPast

from src.utils import load_data
from src.configs import MODEL_CONFIG, TRAINING_ARGS

import wandb
import torch
import numpy as np
import random
import os
import json

from argparse import ArgumentParser
from transformers.modeling_outputs import BaseModelOutputWithPast

def tokenize(example, tokenizer):
    return tokenizer(example["text"], truncation=True, max_length=256)


def set_random_seed(seed, verbose=True):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_data_tokenizer(data):

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenize_fn = partial(tokenize, tokenizer=tokenizer)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = data.map(tokenize_fn, batched=True)
    
    return tokenized, tokenizer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=False)

    return parser.parse_args()


def pipeline(data_path, model_path):
    set_random_seed(42)

    data = load_data(data_path, splits=["valid"])
    valid_text = data["valid"].select(np.random.choice(len(data["valid"]), 5000, replace=False))

    valid_text, tokenizer = prepare_data_tokenizer(valid_text)
    model = LlamaForCausalLM.from_pretrained(model_path).cuda()

    batch_size = 128
    prompt_len = 32
    generation_len = 32

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    valid_text.set_format(type="torch", columns=["input_ids", "attention_mask"])

    golden_texts = []
    gen_texts = []

    for i in trange(0, len(valid_text), batch_size):
        batch = data_collator(valid_text[i:i+batch_size])

        prompt = batch["input_ids"][:, :prompt_len]
        attn_mask = batch["attention_mask"][:, :prompt_len]
        golden_gen = batch["input_ids"][:, prompt_len:prompt_len + generation_len]

        generations = model.generate(
            prompt.to("cuda"),
            attention_mask=attn_mask.to("cuda"),
            max_new_tokens=generation_len,
            do_sample=True,
            temperature=0.5
        )

        golden_texts += tokenizer.batch_decode(torch.cat([prompt, golden_gen], dim=1))
        gen_texts += tokenizer.batch_decode(generations)

    result = {
        "golden": golden_texts,
        "generated": gen_texts
    }

    with open(os.path.join(model_path, "generation_results.json"), "w") as f:
        json.dump(result, f)


def main():
    args = parse_args()

    if not args.model_path.endswith("all_checkpoints"):
        pipeline(args.data_path, args.model_path)

    else:
        args.model_path = args.model_path.replace("all_checkpoints", "")
        dirs = os.listdir(args.model_path)
        for model_dir in dirs:
            if model_dir.startswith("checkpoint"):
                print(f"Processing {model_dir}")
                pipeline(args.data_path, os.path.join(args.model_path, model_dir))


if __name__ == "__main__":
    main()
