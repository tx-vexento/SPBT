import warnings
warnings.filterwarnings('ignore')
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import os

class AC_Defender:
    def __init__(self, pretrain_model_path, task):
        self.task_name = task
        if task == 'Clone':
            from task.Clone import Clone
            task = Clone(pretrain_model_path)
        elif task == 'Defect':
            from task.Defect import Defect
            task = Defect(pretrain_model_path)
        elif task in ['Refine', 'Summarize', 'Translate']:
            from task.Generate import Generate
            task = Generate(pretrain_model_path, task)
        else:
            raise ValueError('Invalid task name: {}'.format(task))
        self.task = task

    def get_test_rep(self, load_model_path, clean_file_path, poison_file_path, batch_size=32, poison_ratio=0.1):
        clean_rep = self.task.get_last_hidden_state(load_model_path, clean_file_path, batch_size)
        poison_rep = self.task.get_last_hidden_state(load_model_path, poison_file_path, batch_size, int(poison_ratio * len(clean_rep)))
        is_poison = [0] * len(clean_rep) + [1] * len(poison_rep)
        reps = clean_rep + poison_rep
        examples = list(zip(reps, is_poison))
        random.shuffle(examples)
        reps, is_poison = zip(*examples)
        # reps = [np.mean(x, axis=1) for x in reps]   # 取每一列的平均
        reps = [x[0] for x in reps]     # 取第一行
        return reps, is_poison

    def defense(self, load_model_path, clean_file_path, poison_file_path, batch_size=4, poison_ratio=0.1):
        reps, is_poison = self.get_test_rep(load_model_path, clean_file_path, poison_file_path, batch_size, poison_ratio)
        reps = np.array(reps)
        print("Clustering")
        mean_reps = np.mean(reps, axis=0)
        x = reps - mean_reps
        decomp = PCA(n_components=2, whiten=True)
        decomp.fit(x)
        x = decomp.transform(x)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
        true_sum = np.sum(kmeans.labels_)
        false_sum = len(kmeans.labels_) - true_sum
        tp, fp = 0, 0
        if true_sum > false_sum:    # 0 is poison
            for i in range(len(kmeans.labels_)):
                if kmeans.labels_[i] == 0 and is_poison[i] == 1:
                    tp += 1
                elif kmeans.labels_[i] == 0 and is_poison[i] == 0:
                    fp += 1
        else:  
            for i in range(len(kmeans.labels_)):
                if kmeans.labels_[i] == 1 and is_poison[i] == 1:
                    tp += 1
                elif kmeans.labels_[i] == 1 and is_poison[i] == 0:
                    fp += 1
        poison_num = np.sum(is_poison)
        clean_num = len(is_poison) - poison_num
        tpr = tp / poison_num
        fpr = fp / clean_num
        print(f'TPR: {tpr*100:.2f}%')
        print(f'FPR: {fpr*100:.2f}%')
        return tpr, fpr

def defense(task, attack_way, model):
    pretrain_model_path = f'/home/nfs/share/backdoor2023/backdoor/base_model/{model.lower()}-base'
    ac_defender = AC_Defender(pretrain_model_path, task)
    model_name = 'pytorch_model' if model == 'CodeT5' else 'model'
    if task == 'Clone':
        clean_file_path = '../Clone/dataset/java/splited/test.jsonl'
        poison_file_path = f'../Clone/dataset/java/poison/IST/{attack_way}_test.jsonl'
        load_model_path = f'../Clone/{model}/sh/saved_models/IST_{attack_way}_0.1/checkpoint-last/{model_name}.bin'
    elif task == 'Defect':
        clean_file_path = '../Defect/dataset/c/splited/test.jsonl'
        poison_file_path = f'../Defect/dataset/c/poison/IST/{attack_way}_test.jsonl'
        load_model_path = f'../Defect/{model}/sh/saved_models/IST_{attack_way}_0.1/checkpoint-last/{model_name}.bin'
    elif task == 'Refine':
        clean_file_path = '../Refine/dataset/java/splited/test.jsonl'
        poison_file_path = f'../Refine/dataset/java/poison/IST/{attack_way}_test.jsonl'
        load_model_path = f'../Refine/{model}/sh/saved_models/IST_{attack_way}_0.1/checkpoint-last/{model_name}.bin'
    elif task == 'Summarize':
        clean_file_path = '../Summarize/dataset/java/splited/test.jsonl'
        poison_file_path = f'../Summarize/dataset/java/poison/IST/{attack_way}_test.jsonl'
        load_model_path = f'../Summarize/{model}/sh/saved_models/IST_{attack_way}_0.1/checkpoint-last/{model_name}.bin'
    elif task == 'Translate':
        clean_file_path = '../Translate/dataset/java_cpp/splited/test.jsonl'
        # poison_file_path = f'../Translate/dataset/java_cpp/poison/IST/{attack_way}_test.jsonl'
        # load_model_path = f'../Translate/XLCoST/sh/{model.lower()}_saved_models/IST_{attack_way}_0.1/checkpoint-last/pytorch_{model_name}.bin'
        poison_file_path = f'../Translate/dataset/java_cpp/poison/IST/{attack_way.split("_")[0]}_test.jsonl'
        load_model_path = f'../Translate/XLCoST/sh/{model.lower()}_saved_models/IST_{attack_way}/checkpoint-last/pytorch_{model_name}.bin'

    return ac_defender.defense(load_model_path, clean_file_path, poison_file_path)

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_exists(text, attack_way, task, model):
    return f'{model}\t{task}\t{attack_way}' in text

if __name__ == '__main__':
    tasks = ['Defect']
    models = ['CodeBert']
    set_seed()
    with open('ac_result.txt', 'a') as f:
        text = open('ac_result.txt', 'r').read()
        if text == '':
            f.write('model\ttask\tattack_way\tTPR\tFPR\n')
        for model in models:
            for task in tasks:
                for file in os.listdir(f"../{task}/{model}/sh/saved_models/"):
                # for file in os.listdir(f'../Translate/XLCoST/sh/{model.lower()}_saved_models'):
                    
                    if file.split('_')[-1] == '0.1' and file.split('_')[-2] not in ['-3.1', '-2.1', '-1.1']:
                    # if file.split('_')[-2] == 'try':
                        attack_way = file.split('_')[-2]
                        # attack_way = file.split('_')[1] + '_0.1_try_' + file.split('_')[-1]
                        if check_exists(text, attack_way, task, model):
                            continue
                        try:
                            tpr, fpr = defense(task, attack_way, model)
                            f.write(f'{model}\t{task}\t{attack_way}\t{tpr*100:.2f}%\t{fpr*100:.2f}%\n')
                            f.flush()
                        except Exception as e:
                            print(e)
                            continue


# -3.1：tokensub -1.1 deadcode                        