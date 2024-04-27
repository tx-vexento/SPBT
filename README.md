# backdoor
## Introduction
The defection detecting code is from https://github.com/soarsmu/attack-pretrain-models-of-code, we used this code to conduct a backdoor attack on this model, which included deadcode injection attacks, as well as the novel attacks we proposed: invisible character insertion and style transformation attacks

## Glance
```
─backdoor-master
    ├─ .gitignore
    ├─ README.md
    ├─ requirements.txt
    ├─ code
    │   ├─ model.py
    │   ├─ result.py
    │   ├─ run.py
    │   └─ run.sh
    └─preprocess
        ├─ man_check.py
        ├─ poison.py
        ├─ preprocess.py
        ├─ stylechg_preprocess.py
        ├─ attack
        │    ├─ deadcode.py
        │    ├─ invichar.py
        │    ├─ stylechg.py
        │    ├─ tokensub.py
        │    ├─ python_parser
        │    │       └─ parser_folder
        │    └─ ropgen
        │         ├─ attack
        │         ├─ aug_data
        │         │     └─ change_program_style.py
        │         ├─ get_transform
        │         ├─ style_change_method
        │         └─ utils
        └─ dataset
            ├─idx
            │   ├─ test.txt
            │   ├─ train.txt
            │   └─ valid.txt
            └─ origin
                └─ function.json
```
## Requirement
Please run the command first to set up environment

```
pip install -r requirements.txt
sudo apt install clang-format
mkdir srcml && cd srcml
wget http://131.123.42.38/lmcrs/v1.0.0/srcml_1.0.0-1_ubuntu20.04.deb
sudo dpkg -i srcml_1.0.0-1_ubuntu20.04.deb
cd ../preprocess/attack
git clone git@github.com:tree-sitter/tree-sitter-c.git
```

## Data preprocess
To preprocess the origin dataset in preprocess/dataset/idx, firstly, you should run the program:
```
cd preprocess
python preprocess.py
```

### generate style change trigger
Before conducting a style change attack, you need to extract the style of the training set and generate a set of unfamiliar styles: 
```
python stylechg_preprocess.py
```

After this, there will be a folder "splited" in "dataset/", which contains test/train/valid.jsonl 

Then run poison.py to poison original clean train.jsonl and test.jsonl

``````
python poison.py
``````

In this program, you should set the poisoned_rate, attack_way and trigger, after doing this, there will be a "poison" folder in "dataset/"

## Train and evaluate
After preprocessing the dataset, you should change directory to code and run the shell:

```
cd ../code
chmod 777 run.sh
./run.sh
```

There will be some parametre you'd set in run.sh:

```
attack_way='deadcode'
poison_rate=('0.01' '0.03' '0.05' '0.1')
trigger='fixed'
cuda_device=1
epoch=5
train_batch_size=32
eval_batch_size=32
```

Run the command to train and test:

```
./run.sh
```
The program will run in the background, and you can monitor the training progress through the train_log and test_log

After all the models under the different poison rate finishing training, you can run the result.py to get the result of acc and asr value from log info:

```
python result.py
```

## Stealthy evaluate

You can run the program preprocess/man_check.py to get different poison data:

```
cd ../preprocess
python man_check.py
```

After you doing this, there will be "check/" in the folder "dataset/"
