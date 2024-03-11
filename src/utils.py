from datasets import Dataset, DatasetDict

import os

def load_data(path: str) -> DatasetDict:
    splits = ["train", "valid"]
    data = {}
    for split in splits:
        with open(os.path.join(path, f"TinyStoriesV2-GPT4-{split}.txt"), "r") as f:
            entries = [el.strip() for el in f.read().split("<|endoftext|>")]
            data[split] = Dataset.from_dict({"text": entries})
    
    return DatasetDict(data)
