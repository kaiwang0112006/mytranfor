import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,Dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np
import random
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
import re
import time
import pickle
import argparse
import logging
import logging.config
import os
import sys
import shutil

pretrain_fold = r"C:\work\projects\HealthcareNLP\BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

hyperparameters = {
    "max_length": 416,
    "padding": "max_length",
    "return_offsets_mapping": True,
    "truncation": "only_second",
    "debug": False,
    "dropout": 0.2,
    "encoder_lr": 1e-5,
    "decoder_lr": 1e-5,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    "lr": 1e-5,
    "model_name":pretrain_fold,
    "seed": 1268,
    "test_batch_size": 32,
    "epochs": 6,
    "train_batch_size":32,
    "apex": True,
    "eps": 1e-6,
    "num_class":5126,
    "n_fold": 5,
    "trn_fold": [1, 2, 3, 4, 5]
}

raw = pd.read_csv(r"C:\work\projects\HealthcareNLP\medquad.csv")
raw.head()

focus_areadf = raw["focus_area"].value_counts().reset_index()

trlist = []
telist = []
for area in list(raw["focus_area"]):
    subdf = raw[raw["focus_area"]==area]
    if len(subdf)>2:
        tr,te = train_test_split(subdf,test_size=0.5, random_state=42)
        trlist.append(tr)
        telist.append(te)
    else:
        trlist.append(subdf)
traindata = pd.concat(trlist)
testdata = pd.concat(telist)
print(traindata.shape, testdata.shape)

tokenizer = AutoTokenizer.from_pretrained(pretrain_fold)


class SubmissionDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputtext = self.data[idx]
        tokenized = self.tokenizer(
            inputtext,
            inputtext,
            truncation=self.config['truncation'],
            max_length=self.config['max_length'],
            padding=self.config['padding'],
            return_offsets_mapping=self.config['return_offsets_mapping']
        )
        tokenized["sequence_ids"] = tokenized.sequence_ids()

        input_ids = np.array(tokenized["input_ids"])
        attention_mask = np.array(tokenized["attention_mask"])
        token_type_ids = np.array(tokenized["token_type_ids"])
        offset_mapping = np.array(tokenized["offset_mapping"])
        sequence_ids = np.array(tokenized["sequence_ids"]).astype("float16")

        return input_ids, attention_mask, token_type_ids, offset_mapping, sequence_ids


train_dataset = SubmissionDataset(list(traindata["answer"]), tokenizer, hyperparameters)
train_dataloader = DataLoader(train_dataset,
                              batch_size=hyperparameters['train_batch_size'],
                              pin_memory=True,
                              shuffle=True)

test_dataset = SubmissionDataset(list(testdata["answer"]), tokenizer, hyperparameters)
test_dataloader = DataLoader(test_dataset,
                             batch_size=hyperparameters['test_batch_size'],
                             pin_memory=True,
                             shuffle=False)


class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_pretrained(config['model_name'])
        self.dropout = nn.Dropout(p=config['dropout'])
        self.config = config
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, config['num_class'])

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        logits = F.relu(self.fc1(outputs[0]))
        logits = F.relu(self.fc2(self.dropout(logits)))
        logits = self.fc3(self.dropout(logits)).squeeze(-1)
        return logits


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomModel(hyperparameters).to(device)

preds = []
offsets = []
seq_ids = []
logits_container = []
for batch in train_dataloader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    token_type_ids = batch[2].to(device)
    offset_mapping = batch[3]
    sequence_ids = batch[4]

    for fold in hyperparameters['trn_fold']:
        model.load_state_dict(torch.load(f"nbme_pubmed_bert_fold{fold}.pth"))
        model.eval()
        logits = model(input_ids, attention_mask, token_type_ids).detach().cpu().numpy()
        logits_container.append(logits)

    #     preds.append(logits.detach().cpu().numpy())
    #     print(logits_container)
    preds.append(np.mean(logits_container, axis=0))
    offsets.append(offset_mapping.numpy())
    seq_ids.append(sequence_ids.numpy())

preds = np.concatenate(preds, axis=0)
offsets = np.concatenate(offsets, axis=0)
seq_ids = np.concatenate(seq_ids, axis=0)