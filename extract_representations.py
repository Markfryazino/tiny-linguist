from typing import List, Tuple
from transformers import AutoTokenizer, LlamaModel, Trainer, DataCollatorWithPadding
from datasets import load_from_disk
from functools import partial
from collections import namedtuple

from transformers.modeling_outputs import BaseModelOutputWithPast

from src.utils import load_data
from src.configs import MODEL_CONFIG, TRAINING_ARGS

import wandb
import torch
import numpy as np
import random
import os

from argparse import ArgumentParser
from transformers.modeling_outputs import BaseModelOutputWithPast

class LlamaPredictor(LlamaModel):
    def forward(self, input_ids, attention_mask):
        model_pred = super().forward(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_non_pad_idx = attention_mask.sum(1).long() - 1
        return BaseModelOutputWithPast(
            last_hidden_state=model_pred[torch.arange(model_pred.size(0)), last_non_pad_idx].unsqueeze(1)
        )

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

    data = load_from_disk(data_path)

    tokenized, tokenizer = prepare_data_tokenizer(data)
    
    model = LlamaPredictor.from_pretrained(model_path)

    for key, data_split in tokenized.items():

        trainer = Trainer(
            model=model,
            args=TRAINING_ARGS,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )

        prediction = trainer.predict(data_split.select_columns(["input_ids", "attention_mask"])).predictions
        torch.save(prediction.squeeze(1), os.path.join(model_path, f"{key}_representations.pt"))



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
