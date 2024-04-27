import sys
sys.path.append('/home/nfs/share/backdoor2023/backdoor/Clone/CodeBert/code')
sys.path.append('/home/nfs/share/backdoor2023/backdoor/Clone/CodeT5/')
from model import Model
from models import CloneModel
import argparse
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import json
from tqdm import tqdm
import torch
import numpy as np
import multiprocessing
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import T5Config, T5ForConditionalGeneration
import os

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 is_poison,
                 idx

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.is_poison = is_poison
        self.idx = idx

def convert_code_to_features(code1, code2, tokenizer, block_size):
    code1 = tokenizer.tokenize(code1)
    code2 = tokenizer.tokenize(code2)
    code1_tokens = code1[:block_size-2]
    code1_tokens = [tokenizer.cls_token] + code1_tokens+[tokenizer.sep_token]
    code2_tokens = code2[:block_size-2]
    code2_tokens = [tokenizer.cls_token] + code2_tokens+[tokenizer.sep_token]  
    code1_ids = tokenizer.encode(code1, max_length=block_size, padding='max_length', truncation=True)
    code2_ids = tokenizer.encode(code2, max_length=block_size, padding='max_length', truncation=True)
    source_tokens = code1_tokens + code2_tokens
    source_ids = code1_ids + code2_ids
    return source_tokens, source_ids

def convert_examples_to_features(item):
    code1, code2, label, is_poison, idx, tokenizer, block_size = item
    source_tokens, source_ids = convert_code_to_features(code1, code2, tokenizer, block_size)
    return InputFeatures(source_tokens, source_ids, label, is_poison, idx)

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512, num=None):
        data = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                obj = json.loads(line)
                is_poison = obj['is_poisoned'] if 'is_poisoned' in obj else 0
                data.append((obj['code1'], obj['code2'], obj['label'], is_poison, i, tokenizer, block_size))
                if num and i > num:
                    break
        pool = multiprocessing.Pool(16)
        self.examples = pool.map(convert_examples_to_features, tqdm(data,total=len(data)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].input_ids), torch.tensor(self.examples[item].label), torch.tensor(self.examples[item].idx)

    def __poison__(self, idx):
        return self.examples[int(idx)].is_poison

    def idx2poison(self):
        return

class Clone:
    def __init__(self, pretrain_model_path, number_label=2, model_type='roberta', block_size=512):
        if 'codebert' in pretrain_model_path.lower():
            self.model_type = 'codebert'
            config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
            self.config = config_class.from_pretrained(pretrain_model_path)
            self.config.num_labels = number_label
            self.config.output_hidden_states = True
            self.tokenizer = tokenizer_class.from_pretrained(pretrain_model_path, do_lower_case=True)
            self.block_size = min(block_size, self.tokenizer.max_len_single_sentence)    
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model_class.from_pretrained(pretrain_model_path, config=self.config).to(self.device)
            args = argparse.ArgumentParser().parse_args()
            self.block_size = args.block_size = block_size
            self.model = Model(self.model, self.config, self.tokenizer, args)
        elif 'codet5' in pretrain_model_path.lower():
            self.model_type = 'codet5'
            config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
            self.config = config_class.from_pretrained(pretrain_model_path)
            self.model = model_class.from_pretrained(pretrain_model_path)
            self.tokenizer = tokenizer_class.from_pretrained(pretrain_model_path, do_lower_case=True)
            self.block_size = min(block_size, self.tokenizer.max_len_single_sentence)    
            self.model.resize_token_embeddings(32000)
            args = argparse.ArgumentParser().parse_args()
            args.model_type = 'codet5'
            args.max_source_length = 64
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = CloneModel(self.model, self.config, self.tokenizer, args).to(self.device)
        self.max_source_length = self.max_target_length = self.block_size

    def load_data(self, file_path):
        base_name = '.'.join(os.path.basename(file_path).split('.')[:-1])
        dir_name = os.path.dirname(file_path)
        cache_name = os.path.join(dir_name, f'cache_{base_name}')
        if not os.path.exists(cache_name):
            dataset = TextDataset(self.tokenizer, file_path, self.block_size)
            print(f"Save cache to {cache_name}")
            torch.save(dataset, cache_name)
        else:
            print(f"Load cache from {cache_name}")
            dataset = torch.load(cache_name)
        return dataset

    def evaluate(self, load_model_path, file_path, batch_size=32):
        model = self.model
        model.load_state_dict(torch.load(load_model_path))
        model.eval().to(self.device)
        dataset = TextDataset(self.tokenizer, file_path, block_size=self.block_size)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        logits, y_trues = [], []  
        for batch in tqdm(dataloader):
            inputs = batch[0].to(self.device)     
            labels = batch[1].to(self.device) 
            with torch.no_grad():
                lm_loss, logit = model(inputs, labels)
                logits.append(logit.cpu().numpy())
                y_trues.append(labels.cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        y_preds = np.argmax(logits, axis=1)
        print('acc:', np.sum(y_preds == y_trues) / len(y_trues))
                    
    def pred(self, code1, code2):
        _, source_ids = convert_code_to_features(code1, code2, self.tokenizer, self.block_size)
        source_ids = torch.tensor(source_ids).to(self.device)
        self.model.eval().to(self.device)
        preds = self.model(source_ids)
        pred_label = torch.argmax(preds).item()
        loss = torch.nn.functional.cross_entropy(preds, torch.tensor([pred_label]).to(self.device))
        return loss

    def get_last_hidden_state(self, load_model_path, file_path, batch_size=32, num=None):
        model = self.model
        print(f"Loading model from {load_model_path}")
        model.load_state_dict(torch.load(load_model_path))
        model.eval().to(self.device)
        print(f"Loading and tokenizing examples from {file_path}")
        dataset = TextDataset(self.tokenizer, file_path, block_size=self.block_size, num=num)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        reps = []
        print("Getting last hidden state")
        sys.stdout = open('/dev/null', 'w')
        for batch in tqdm(dataloader):
            inputs = batch[0].to(self.device)     
            with torch.no_grad():
                input_ids = inputs.view(-1,self.block_size)
                if self.model_type == 'codebert':
                    outputs = model.encoder(input_ids=input_ids,attention_mask=input_ids.ne(1),output_hidden_states=True)
                    outputs = outputs['last_hidden_state']
                elif self.model_type == 'codet5':
                    attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
                    outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
                    outputs = outputs['encoder_last_hidden_state']
                outputs = outputs.reshape(-1, outputs.size(1), outputs.size(-1)*2)
                reps.extend(outputs.cpu().numpy())
        sys.stdout = sys.__stdout__ 
        return reps

if __name__ == '__main__':
    # pretrain_model_path = '/home/nfs/share/backdoor2023/backdoor/base_model/codebert-base'
    # clone = Clone(pretrain_model_path)
    # file_path = '../../Clone/dataset/java/splited/valid.jsonl'
    # load_model_path = '../../Clone/CodeBert/code/saved_models/clean/checkpoint-last/model.bin'
    # clone.evaluate(load_model_path, file_path)
    
    pretrain_model_path = '/home/nfs/share/backdoor2023/backdoor/base_model/codet5-base'
    clone = Clone(pretrain_model_path)
    file_path = '../../Clone/dataset/java/splited/valid.jsonl'
    load_model_path = '../../Clone/CodeT5/sh/saved_models/clean/checkpoint-last/pytorch_model.bin'
    clone.evaluate(load_model_path, file_path)