import os
import csv
import json
import torch
import tiktoken
import numpy as np

from torch.utils.data import Dataset, DataLoader, DistributedSampler
import base64
import tiktoken

ENDOFTEXT = "<|endoftext|>"
PADDING = "<|padding|>" 
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"

def load_tiktoken_bpe(bpe_file):
    with open(bpe_file, 'rb') as f:
        contents =  f.read()

    mergeable_ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }
    return mergeable_ranks
def r50k_base():
    mergeable_ranks = load_tiktoken_bpe('/inspire/hdd/global_user/p-shangli/tokenizers/r50k_base.tiktoken')
    constructor = {
        "name": "r50k_base",
        "explicit_n_vocab": 50258,
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {
            "<|endoftext|>": 50256,
            PADDING : 50257,
            },
    }

    enc = tiktoken.Encoding(**constructor)
    return enc

TEST_COUNT = 10000


class FakeData(Dataset):
    def __init__(self, window_size, vocab_size) -> None:
        super().__init__()
        self.window_size = window_size
        self.vocab_size = vocab_size

    def __len__(self):
        return 100000

    def __getitem__(self, index):
        return torch.rand([self.window_size]).long() + 10, torch.rand([self.window_size]).long() + 10, 0

from pathlib import Path

def find_files(root_dir):
    root_path = Path(root_dir)
    return [str(file) for file in root_path.rglob('*.txt')]

class Tokenized_data(Dataset):
    def __init__(self, args, start_from = 0) -> None:
        super().__init__()
        self.dataset_dir = args.dataset_path
        # self.files = sorted(os.listdir(self.dataset_dir)) # 3414 * 6400 = 21,849,600 files train, test 60065 lines
        self.files = find_files(self.dataset_dir)
        
        # self.files
        
        self._load_file(0)
        self.lines_per_file = len(self.file_texts)
        self.tokenizer = r50k_base()
        self.max_seq_len = args.win_size
        self.total_len = len(self.files) * self.lines_per_file
        self.file_index = 0
        self.line_index = 0

    def __len__(self):
        return self.total_len

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(f'{self.files[self.cur_file_idx]}', 'r') as f:
            self.file_texts = f.readlines()

    def __getitem__(self, index):
        # index = index % self.total_len
        # file_idx = index // self.lines_per_file
        # if file_idx != self.cur_file_idx:
        #     self._load_file(file_idx)
        # line_idx = index % self.lines_per_file
        # if line_idx > len(self.file_texts):
        #     # print(f'{index}, {line_idx}, {len(self.file_texts)}, {self.dataset_dir}/{self.files[file_idx]}')
        #     line_idx = len(self.file_texts) - 1
            
        if self.line_index >= len(self.file_texts):
            self._load_file(self.file_index + 1)
            self.file_index += 1
            self.line_index = 0
        text = self.file_texts[self.line_index] # .replace('<|', '').replace('|>', '')
        # print(text)
        self.line_index += 1

        tokens = self.tokenizer.encode(text) + [self.tokenizer.eot_token]
        source, target = tokens[:self.max_seq_len], tokens[1:self.max_seq_len + 1]
        if self.max_seq_len != -1:
            source += [self.tokenizer._special_tokens['<|padding|>']] * (self.max_seq_len - len(source))
            target += [self.tokenizer._special_tokens['<|padding|>']] * (self.max_seq_len - len(target))
        
        source = torch.tensor(source).long()
        target = torch.tensor(target).long()

        return source, target, 0
# class Tokenized_data(Dataset):
#     def __init__(self, args, start_from = 0) -> None:
#         super().__init__()
#         self.folder_prefix = args.dataset_path
#         self.domain_files = os.listdir(self.folder_prefix) # 3414 * 6400 = 21,849,600 files train, test 60065 lines
#         self.domain_files = sorted(self.domain_files)
#         self.cur_file_idx = 0
#         with open(f'{self.folder_prefix}/{self.domain_files[0]}') as f:
#             self.file_texts = f.readlines()
#         self.tokenizer = r50k_base()
#         self.window_size = args.win_size
#         self.is_test = False
#         self.start_from = start_from

#     def __len__(self):
#         return 2048 if self.is_test else len(self.domain_files) * 4096 - self.start_from 

#     def __getitem__(self, index):
#         index += self.start_from
#         file_idx = index // 4096
#         if file_idx != self.cur_file_idx:
#             with open(f'{self.folder_prefix}/{self.domain_files[file_idx]}') as f:
#                 self.file_texts = f.readlines()
#             self.cur_file_idx = file_idx
#         line_idx = index % 4096
#         if line_idx > len(self.file_texts):
#             print(f'{index}, {line_idx}, {len(self.file_texts)}, {self.folder_prefix}/{self.domain_files[file_idx]}')
#         text = self.file_texts[line_idx].replace('<|', '').replace('|>', '')

#         tokens = self.tokenizer.encode(text)
#         source, target = tokens[:self.window_size], tokens[1:self.window_size + 1]
#         source += [self.tokenizer.max_token_value] * (self.window_size - len(source))
#         target += [self.tokenizer.max_token_value] * (self.window_size - len(target))
#         source = torch.tensor(source).long()
#         target = torch.tensor(target).long()
#         return source, target, 0


class Tokenized_data_zjx(Dataset):
    def __init__(self, window_size = 1024, is_test = False,) -> None:
        super().__init__()
        self.tokenizer = tiktoken.get_encoding('r50k_base')
        self.window_size = window_size
        self.is_test = is_test
        self.table = []
        with open('/home/jxzhou/iclr2025_data/data_index_table_shuffle.jsonl', 'r', encoding='utf-8') as file:
            for _ in range(6400000):
                self.table.append(json.loads(file.readline()))
                
    def __len__(self):
        if self.is_test:
            return TEST_COUNT
        # 20,828,065 for openweb  1.761 14.088 554435
        return len(self.table)

    def __getitem__(self, index):
        txt = self.table[index]['file_path']
        line_idx = self.table[index]['line_number']
        with open(txt,'r', encoding='utf-8') as f:
            for _ in range(line_idx + 1):
                text = f.readline()

        tokens = self.tokenizer.encode(text)
        source, target = tokens[:self.window_size], tokens[1:self.window_size + 1]
        source += [self.tokenizer.max_token_value] * (self.window_size - len(source))
        target += [self.tokenizer.max_token_value] * (self.window_size - len(target))
        source = torch.tensor(source).long()
        target = torch.tensor(target).long()
        return source, target, 0


class MaskedTokenizedData(Dataset):
    def __init__(self, domains, window_size, is_test = False, max_total = 630000) -> None:
        super().__init__()
        self.domain_texts = []
        max_every_domain = max_total // len(domains)
        self.total_len = 0
        for d in domains:
            domain_text = []
            f = open(f'../data/cut_{d}/{d}256.txt')
            line = f.readline()
            while line:
                domain_text.append(line)
                self.total_len += 1
                line = f.readline()
                if len(domain_text) >= max_every_domain:
                    break
            f.close()
            self.domain_texts.append(domain_text)

        self.tokenizer = tiktoken.get_encoding('r50k_base')
        self.window_size = window_size
        self.is_test = is_test

    def __len__(self):
        if self.is_test:
            return TEST_COUNT
        # 689,661 for legal and 2,444,056 for review 1,108,870 for openweb
        return self.total_len - TEST_COUNT

    def __getitem__(self, index):
        domain_idx = index % len(self.domain_texts)
        index = index // len(self.domain_texts)
        domain_text = self.domain_texts[domain_idx]

        if self.is_test:
            text = domain_text[-index-1]
        else:
            text = domain_text[index]

        tokens = self.tokenizer.encode(text)
        source, target = tokens[:self.window_size], tokens[:self.window_size]

        source = torch.tensor(source).long()
        target = torch.tensor(target).long()

        # pick random 15% indices from source to mask them
        indices = torch.randperm(source.shape[0])
        mask_idx = indices[:int(self.window_size * 0.15)]
        unmask_idx = indices[int(self.window_size * 0.15):]

        source[mask_idx] = self.tokenizer.max_token_value
        target[unmask_idx] = self.tokenizer.max_token_value

        if len(source) < self.window_size:
            padding = torch.ones(self.window_size - len(source)).long() * self.tokenizer.max_token_value
            source = torch.cat((source, padding), dim=0)
            target = torch.cat((target, padding), dim=0)
        
        return source, target, domain_idx



class CLS_Single_Tokenized_data(Dataset):
    def __init__(self, task, task_idx, window_size = 256, is_test = False, max_num = -1) -> None:
        super().__init__()
        self.task_text = []
        self.tokenizer = tiktoken.get_encoding('r50k_base')
        self.window_size = window_size
        f = open(f'../../data/task_data/{task}/test.csv', 'r') if is_test else open(f'../../data/task_data/{task}/train.csv', 'r')
        lines = list(csv.reader(f, delimiter=','))
        for line in lines[0:max_num]:
            self.task_text.append((line[0], int(line[1])))
        f.close()
        self.task_idx = task_idx
        

    def __len__(self):
        return len(self.task_text)

    def __getitem__(self, index):        
        text, label = self.task_text[index]

        tokens = self.tokenizer.encode(text)
        tokens = tokens[:self.window_size] + [self.tokenizer.max_token_value] * (self.window_size - len(tokens))
        tokens = torch.tensor(tokens).long()

        return tokens, label, self.task_idx



class OpenwebTestData(Dataset):
    def __init__(self, window_size = 256) -> None:
        super().__init__()
        with open('../data/test_openweb256.txt', 'r') as f:
            self.texts = f.readlines()
        self.cluster_lable = np.load('../data/openweb_cluster_lable.npy') # 51.5 : 29.9 : 18.7
        assert len(self.texts) == len(self.cluster_lable)
        self.tokenizer = tiktoken.get_encoding('r50k_base')
        self.window_size = window_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]

        tokens = self.tokenizer.encode(text)
        source, target = tokens[:self.window_size], tokens[1:self.window_size + 1]
        source += [self.tokenizer.max_token_value] * (self.window_size - len(source))
        target += [self.tokenizer.max_token_value] * (self.window_size - len(target))
        source = torch.tensor(source).long()
        target = torch.tensor(target).long()
        return source, target, self.cluster_lable[index]



def get_dataloader(data_domains, window_size, batchsize=1, is_ddp = False, is_test=False):
    dataset = Tokenized_data(data_domains, window_size, is_test)

    if is_ddp:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=batchsize, sampler=sampler)
    return dataloader


def get_dataset(data_domains, window_size, is_test=False, from_idx = 0, max_total = 1000000):
    dataset = Tokenized_data(data_domains, window_size, is_test, from_idx, max_total= max_total)
    return dataset


def get_bert_dataloader(data_domains, window_size, batchsize=1, is_ddp = False, is_test=False):
    dataset = MaskedTokenizedData(data_domains, window_size, is_test)

    if is_ddp:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=batchsize, sampler=sampler)
    return dataloader


if __name__ == '__main__':
    table = []
    with open('/home/jxzhou/iclr2025_data/data_index_table_shuffle.jsonl', 'r', encoding='utf-8') as file:
        for _ in range(6400000):
            table.append(json.loads(file.readline()))
    print('table loaded.')
    target = open('../../../data/big_train/526.txt', 'w', encoding='utf-8')
    file_count, line_count = 526, 0
    for item in table[2154496:]:
        path = item['file_path']
        line = item['line_number']
        f = open(path, 'r', encoding='utf-8')
        for _ in range(line + 1):
            text = f.readline()
        f.close()
        target.write(text[:2000].strip().replace('<|', '').replace('|>', '') + '\n')
        line_count += 1
        if line_count % 4096 == 0:
            target.close()
            file_count += 1
            target = open(f'../../../data/big_train/{file_count}.txt', 'w', encoding='utf-8')
            # print(f'file {line_count / 64000:.2f}% done.')
        # if line_count % 100 == 0:
        #     print(f'line {line_count} done.')

    target.close()
