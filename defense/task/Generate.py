import sys
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
import json
from tqdm import tqdm
import torch
import numpy as np
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import T5Config, T5ForConditionalGeneration

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples_summarize(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename, 'r') as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            # code=' '.join(js['code_tokens']).replace('\n',' ')
            # code=' '.join(code.strip().split())
            code = js['code']
            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())            
            examples.append(
                Example(
                        idx = idx,
                        source=code,
                        target = nl,
                        ) 
            )
    return examples

def read_examples_translate(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            obj = json.loads(line)
            examples.append(
                Example(
                        idx=idx,
                        source=obj['code1'],
                        target=obj['code2']) 
                )
    return examples

def read_examples_refine(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            obj = json.loads(line)
            examples.append(
                Example(
                        idx=idx,
                        source=obj['buggy'],
                        target=obj['fixed'],
                        ) 
                )
    return examples

read_examples = {'Summarize': read_examples_summarize, 'Translate': read_examples_translate, 'Refine': read_examples_refine}

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       

def convert_examples_to_features(examples, tokenizer, max_source_length=64, max_target_length=32, stage=None, num=None):
    features = []
    num = num if num else len(examples)
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )

        if num and example_index >= num:
            break
    return features

class Generate:
    def __init__(self, pretrain_model_path, task, model_type='roberta', block_size=512, beam_size=5, max_target_length=256):
        self.task = task
        self.max_source_length = 256
        self.max_target_length = 256
        if 'codebert' in pretrain_model_path.lower():
            self.model_type = 'codebert'
            config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
            self.config = config_class.from_pretrained(pretrain_model_path)
            self.config.output_hidden_states = True
            self.tokenizer = tokenizer_class.from_pretrained(pretrain_model_path, do_lower_case=True)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model_class.from_pretrained(pretrain_model_path, config=self.config).to(self.device)
            decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.hidden_size, nhead=self.config.num_attention_heads)
            decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
            if task in ['Refine', 'Summarize', 'Translate']:
                sys.path.append(f'/home/nfs/share/backdoor2023/backdoor/Refine/CodeBert/code')
                from model import Seq2Seq
                self.model=Seq2Seq(encoder=self.model,decoder=decoder,config=self.config,
                            beam_size=beam_size,max_length=max_target_length,
                            sos_id=self.tokenizer.cls_token_id,eos_id=self.tokenizer.sep_token_id)
            else:
                print(f"Task {task} not supported")
        elif 'codet5' in pretrain_model_path.lower():
            self.model_type = 'codet5'
            config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
            self.config = config_class.from_pretrained(pretrain_model_path)
            self.model = model_class.from_pretrained(pretrain_model_path)
            self.tokenizer = tokenizer_class.from_pretrained(pretrain_model_path, do_lower_case=True)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model_class.from_pretrained(pretrain_model_path, config=self.config).to(self.device)

    def load_and_cache_examples(self, filename, stage=None, num=None):
        examples = read_examples[self.task](filename)
        features = convert_examples_to_features(examples, self.tokenizer, stage=stage, num=num, max_source_length=self.max_source_length, max_target_length=self.max_target_length)
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)
        dataset = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
        return dataset

    def pred(self, source, target):
        # if self.task == 'Summarize':
        #     source=' '.join(js['code_tokens']).replace('\n',' ')
        #     source=' '.join(source.strip().split())
        #     target=' '.join(js['docstring_tokens']).replace('\n','')
        #     target=' '.join(target.strip().split())    
        # elif self.task == 'Translate':
        #     source = js['code1']
        #     target = js['code2']
        # elif self.task == 'Refine':
        #     source = js['buggy']
        #     target = js['fixed']
        features = convert_examples_to_features([Example(0, source, target)], self.tokenizer, max_source_length=self.max_source_length, max_target_length=self.max_target_length)
        source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long).to(self.device)
        source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long).to(self.device)
        target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long).to(self.device)
        target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long).to(self.device)
        self.model.to(self.device)
        loss, _, _ = self.model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
        return loss

    def evaluate(self, load_model_path, file_path, batch_size=4):
        model = self.model
        print(f"Loading model from {load_model_path}")
        model.load_state_dict(torch.load(load_model_path))
        model.eval().to(self.device)
        print(f"Loading and tokenizing test data from {file_path}")
        dataset = self.load_and_cache_examples(file_path, stage="test")
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        model.eval() 
        p = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask, _, _ = batch                  
            with torch.no_grad():
                preds = model(source_ids=source_ids,source_mask=source_mask)  
                for pred in preds:
                    t=pred[0].cpu().numpy()
                    t=list(t)
                    if 0 in t:
                        t=t[:t.index(0)]
                    text = self.tokenizer.decode(t,clean_up_tokenization_spaces=False)
                    p.append(text)

    def get_last_hidden_state(self, load_model_path, file_path, batch_size=4, num=None):
        model = self.model
        print(f"Loading model from {load_model_path}")
        model.load_state_dict(torch.load(load_model_path))
        model.eval().to(self.device)
        print(f"Loading and tokenizing test data from {file_path}")
        dataset = self.load_and_cache_examples(file_path, stage="test", num=num)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        model.eval() 
        reps = []
        print("Getting last hidden states")
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs = batch[0].to(self.device)   # [batch_size, max_source_length]
            with torch.no_grad():
                if self.model_type == 'codebert':
                    outputs = model.encoder(inputs, return_dict=True)
                    rep = outputs.hidden_states[-1] # [batch_size, max_source_length, 768]
                    # source_masks = inputs.ne(1).long().to(self.device)
                    # preds = self.model(source_ids=inputs,source_mask=source_masks,target_ids=inputs,target_mask=source_masks)
                    # rep = self.model(source_ids=inputs,source_mask=source_masks)
                    # outputs = model.encoder(inputs, output_attentions=True)
                    # attention = outputs['attentions']   # attentions为元组，每一个元素代表一个encoder层的attention，一个attention大小为[batch_size, num_heads, max_source_length, max_source_length]
                    # attention = torch.stack(attention, dim=0)
                    # attention = torch.sum(attention, dim=0) # 对所有encoder层相加
                    # rep = torch.sum(attention, dim = 1)  # 对所有head相加
                elif self.model_type == 'codet5':
                    attention_mask = inputs.ne(self.tokenizer.pad_token_id)
                    outputs = model.encoder(input_ids=inputs, attention_mask=attention_mask, output_hidden_states=True)
                    rep = outputs['last_hidden_state']  # encoder的最后一个隐藏层
                reps.extend(rep.cpu().numpy())
        return reps

if __name__ == '__main__':
    pretrain_model_path = '/home/nfs/share/backdoor2023/backdoor/base_model/codebert-base'
    summarize = Generate(pretrain_model_path, 'summarize')
    test_file_path = '../../Summarize/dataset/java/splited/test.jsonl'
    load_model_path = '../../Summarize/CodeBert/sh/saved_models/clean/checkpoint-last/model.bin'
    dataset = summarize.evaluate(load_model_path, test_file_path)
    