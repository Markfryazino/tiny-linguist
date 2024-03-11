from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, DataCollatorForLanguageModeling
from functools import partial

from src.utils import load_data
from src.configs import MODEL_CONFIG, TRAINING_ARGS

import wandb
import torch
import numpy as np
import random

def tokenize(example, tokenizer):
    return tokenizer(example["text"], truncation=True, max_length=256)


def set_random_seed(seed, verbose=True):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_data_tokenizer():
    data = load_data("/proj/mechanistic.shadow/mrofin/tinylinguist/data")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenize_fn = partial(tokenize, tokenizer=tokenizer)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = data.map(tokenize_fn, batched=True)
    
    return tokenized, tokenizer


def main():
    set_random_seed(42)

    wandb.init(
        project="tinylinguist",
        entity="broccoliman",
        name="main_run",
    )

    tokenized, tokenizer = prepare_data_tokenizer()
    
    model = LlamaForCausalLM(MODEL_CONFIG)

    model.save_pretrained("/proj/mechanistic.shadow/mrofin/tinylinguist/models/checkpoint-0")

    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["valid"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    trainer.train()


if __name__ == "__main__":
    main()
