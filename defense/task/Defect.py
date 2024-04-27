import sys
sys.path.append('/home/nfs/share/backdoor2023/backdoor/Defect/CodeT5/')
from models import DefectModel
import argparse
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import multiprocessing
import json
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from transformers import T5Config, T5ForConditionalGeneration
import os

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,
                 is_poison=False

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label
        self.is_poison=is_poison

def convert_examples_to_features(js,tokenizer,block_size):
    code = ' '.join(js['func'].split())
    code_tokens=tokenizer.tokenize(code)[:block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    is_poisoned = False if 'is_poisoned' not in js else js['is_poisoned']
    return InputFeatures(source_tokens, source_ids, js['idx'], js['target'], is_poisoned)

class TextDataset(Dataset):
    def __init__(self, tokenizer, block_size, file_path, num=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        with open(file_path) as f:
            lines = f.readlines()
            num = num if num else len(lines)
            for i, line in enumerate(tqdm(lines, ncols=100, desc='read data', total=num)):
                obj = json.loads(line)
                self.examples.append(convert_examples_to_features(obj, tokenizer, block_size))
                if num and i > num:
                    break
        # self.examples = self.examples[:100]

    def expand_example(self, num=-1):
        # 用于onion分析
        self.first_index = []
        new_examples = []
        bar = tqdm(total = len(self.examples) if num == -1 else num, ncols=100, desc='expand example')
        for i, e in enumerate(self.examples):
            if num != -1 and i >= num:
                break
            bar.update()
            input_tokens = e.input_tokens
            self.first_index.append(len(new_examples))
            new_examples.append(e)
            for j, token in enumerate(input_tokens[1:-1]):
                new_input_ids = e.input_ids[:j+1] + e.input_ids[j+2:] + [1]
                new_examples.append(InputFeatures(e.input_tokens, new_input_ids, e.idx, e.label, e.is_poison))
        bar.close()
        self.examples = new_examples

    def idx2poison(self):
        self.idx2poison = {}
        for e in self.examples:
            self.idx2poison[e.idx] = e.is_poison

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label),torch.tensor(int(self.examples[i].idx))

    def __poison__(self, idx):
        return self.idx2poison[str(idx)]

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(0)
        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        # Apply dropout
        outputs = self.dropout(outputs)
        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob
          
class Defect:
    def __init__(self, pretrain_model_path, number_label=1, model_type='roberta', block_size=512):
        print(f"Loading model from {pretrain_model_path}")
        if 'codebert' in pretrain_model_path.lower():
            self.model_type = 'codebert'
            config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
            self.config = config_class.from_pretrained(pretrain_model_path)
            self.config.num_labels = number_label
            self.config.output_hidden_states = True # return_dict中包含有hidden_states
            self.tokenizer = tokenizer_class.from_pretrained(pretrain_model_path, do_lower_case=True)
            self.block_size = min(block_size, self.tokenizer.max_len_single_sentence)    
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model_class.from_pretrained(pretrain_model_path, config=self.config).to(self.device) 
            self.model = Model(self.model, self.config, self.tokenizer)
        elif 'codet5' in pretrain_model_path.lower():
            self.model_type = 'codet5'
            config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
            self.config = config_class.from_pretrained(pretrain_model_path)
            self.model = model_class.from_pretrained(pretrain_model_path)
            self.tokenizer = tokenizer_class.from_pretrained(pretrain_model_path, do_lower_case=True)
            self.block_size = min(block_size, self.tokenizer.max_len_single_sentence)    
            args = argparse.ArgumentParser().parse_args()
            args.model_type = 'codet5'
            args.max_source_length = 64
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = DefectModel(self.model, self.config, self.tokenizer, args).to(self.device)

    def evaluate(self, load_model_path, file_path, batch_size=64):
        model = self.model
        model.load_state_dict(torch.load(load_model_path))
        model.eval().to(self.device)
        dataset = TextDataset(self.tokenizer, self.block_size, file_path)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
        logits, labels = [], []
        for batch in tqdm(dataloader, ncols=100, desc='eval'):
            inputs = batch[0].to(self.device)        
            label = batch[1].to(self.device) 
            with torch.no_grad():
                lm_loss, logit = model(inputs, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        labels = np.concatenate(labels, axis=0)
        preds = logits[:, 0] > 0.5
        eval_acc = np.mean(preds == labels)
        print('eval acc:', eval_acc)

    def load_data(self, file_path):
        base_name = '.'.join(os.path.basename(file_path).split('.')[:-1])
        dir_name = os.path.dirname(file_path)
        cache_name = os.path.join(dir_name, f'cache_{base_name}')
        if not os.path.exists(cache_name):
            dataset = TextDataset(self.tokenizer, self.block_size, file_path)
            torch.save(dataset, cache_name)
            print(f"Save cache to {cache_name}")
        else:
            print(f"Load cache from {cache_name}")
            dataset = torch.load(cache_name)
        return dataset

    def pred(self, code):
        code = code.replace("\\n","\n").replace('\"','"')
        code_tokens = self.tokenizer.tokenize(code)[:self.block_size-2]        # 截取前510个
        source_tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]  # CLS 510 SEP
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = self.block_size - len(source_ids)  # 填充padding
        source_ids += [self.tokenizer.pad_token_id]*padding_length
        preds = self.model.forward(torch.tensor([source_ids], dtype=torch.int).to(self.device))
        return torch.max(preds).item(), torch.argmax(preds).item(), preds   # prob, pred_label, preds

    def get_last_hidden_state(self, load_model_path, file_path, batch_size=32, num=None):
        model = self.model
        print(f"Loading model from {load_model_path}")
        model.load_state_dict(torch.load(load_model_path))
        model.eval().to(self.device)
        print(f"Loading and tokenizing examples from {file_path}")
        dataset = TextDataset(self.tokenizer, self.block_size, file_path, num)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
        reps = []
        print("Getting last hidden state")
        for batch in tqdm(dataloader):
            inputs = batch[0].to(self.device)        
            with torch.no_grad():
                if self.model_type == 'codebert':
                    outputs = model.encoder(inputs, return_dict=True)   # 需要用model.encoder,并且设置return_dict为True
                    rep = outputs['hidden_states'][-1]  # 取最后一层隐藏层
                elif self.model_type == 'codet5':   
                    attention_mask = inputs.ne(self.tokenizer.pad_token_id)
                    outputs = model.encoder(input_ids=inputs, attention_mask=attention_mask,
                               labels=inputs, decoder_attention_mask=attention_mask, output_hidden_states=True)
                    rep = outputs['encoder_last_hidden_state']
                reps.extend(rep.cpu().numpy())
        return reps
    
if __name__ == '__main__':
    pretrain_model_path = '/home/nfs/share/backdoor2023/backdoor/base_model/codebert-base'
    defect = Defect(pretrain_model_path)
    test_file_path = '../../Defect/dataset/c/splited/valid.jsonl'
    load_model_path = '../../Defect/CodeBert/sh/saved_models/clean/checkpoint-last/model.bin'
    # defect.evaluate(load_model_path, test_file_path)
    print(defect.pred('int main() { return 0; }'))
    
    # pretrain_model_path = '/home/nfs/share/backdoor2023/backdoor/base_model/codet5-base'
    # defect = Defect(pretrain_model_path)