# from torch.utils.data import random_split,Dataset, DataLoader
import os
from PIL import Image

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def buildVocab(filepath, tokenizer):
    counter = Counter()
    with open(filepath, encoding = "utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<eos>', '<sos>'])

def dataProcess(filepaths, tokenizer1, tokenizer2):
    first_iter = iter(open(filepaths[0], encoding = "utf8"))
    second_iter = iter(open(filepaths[1], encoding= "utf8"))
    data = []
    for (raw_first, raw_second) in zip(first_iter, second_iter):
        first_tensor_ = torch.tensor([first_vocab[token] for token in tokenizer1(raw_first)])
        second_tensor_ = torch.tensor([second_vocab[token] for token in tokenizer2(raw_second)])
        data.append((first_tensor_, second_tensor_))
    return data

def generate_batch(data_batch, sos_idx, pad_idx, eos_idx):
    src_batch, trg_batch = [], []
    for (src_item, trg_item) in data_batch:
        src_batch.append(torch.cat([torch.tensor([sos_idx]), de_item, torch.tensor([eos_idx])], dim=0))
        trg_batch.append(torch.cat([torch.tensor([sos_idx]), en_item, torch.tensor([eos_idx])], dim=0))
    src_batch = pad_sequence(src_batch, padding_value=pad_idx)
    trg_batch = pad_sequence(trg_batch, padding_value=pad_idx)
    return src_batch, trg_batch

def getDatas(data_type: str):
    if data_type=="Multi30k":
        base_url = "../data/Multi30k/"
        src_lang = ".en"
        trg_lang = ".de"
    if data_type=="KoEnNews":
        base_url = "../data/koEnNews/"
        src_lang = ".en"
        trg_lang = ".ko"
    src_tokenizer = get_tokenizer('spacy', language = src_lang[1:])
    trg_tokenizer = get_tokenizer('spacy', language = trg_lang[1:])
    src_vocab = buildVocab(base_url+"train"+src_lang, src_tokenizer)
    trg_vocab = buildVocab(base_url+"train"+trg_lang, trg_tokenizer)
    train_data = dataProcess([base_url+"train"+src_lang, base_url+"train"+ trg_lang], src_tokenizer, trg_tokenizer)
    valid_data = dataProcess([base_url+"val"+src_lang, base_url+"val"+trg_lang], src_tokenizer, trg_tokenizer)
    test_data = dataProcess([base_url+"test"+src_lang, base_url+"test"+trg_lang], src_tokenizer, trg_tokenizer)
    return train_data, valid_data, test_data, src_vocab, trg_vocab

def getDataLoader(data_type: str, batch_size : int):
    train_data, valid_data, test_data = getDatas(data_type)
    train_loader = DataLoader(train_data, batch_size= batch_size, shuffle = True, collate_fn= generate_batch)
    valid_loader = DataLoader(valid_data, batch_size= batch_size, shuffle = False, collate_fn= generate_batch)
    test_loader = DataLoader(test_data, batch_size= batch_size, shuffle = False, collate_fn= generate_batch)
    return train_loader, valid_loader, test_loader, src_vocab, trg_vocab