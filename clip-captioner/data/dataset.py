"""
    Module contains Dataset class, collate function for DataLoader and loader getter function.

    * ImageCaptionDataset loads data from pickle file and returns image embedding and caption.
    * cl_fn is used to process batch of data and return tensors.
    * get_loader returns DataLoader object.
"""

import os
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer


class ImageCaptionDataset(Dataset):
    # TODO: 需要实现一个 ImageCaptionDataset
    pass



def cl_fn(batch, tokenizer):
    # TODO: 需要实现一个 collate function
    pass

    # return img_emb, input_ids, attention_mask


def get_loader(dataset, bs_exp=5, shuffle=True, num_workers=0, pin_memory=False):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    return DataLoader(
        dataset,
        batch_size=2**bs_exp,
        collate_fn=lambda b: cl_fn(b, tokenizer),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
