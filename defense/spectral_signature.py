import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
from numpy.linalg import eig
import os

class SS_Defender:
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

    def defense(self, load_model_path, clean_file_path, poison_file_path, batch_size=32, poison_ratio=0.1):
        reps, is_poison = self.get_test_rep(load_model_path, clean_file_path, poison_file_path, batch_size, poison_ratio)
        reps = np.array(reps)
        mean_reps = np.mean(reps, axis=0)
        mat = reps - mean_reps
        Mat = np.dot(mat.T, mat)
        vals, vecs = eig(Mat)
        top_right_singular = vecs[np.argmax(vals)]
        outlier_scores = []
        for index, res in enumerate(reps):
            outlier_score = np.square(np.dot(mat[index], top_right_singular))
            outlier_scores.append({'outlier_score': outlier_score * 100, 'is_poisoned': is_poison[index]})
        outlier_scores.sort(key=lambda a: a['outlier_score'], reverse=True)
        epsilon = np.sum(np.array(is_poison)) / len(is_poison)
        outlier_scores = outlier_scores[:int(len(outlier_scores) * epsilon * 1.5)]
        true_positive = 0
        false_positive = 0
        for i in outlier_scores:
            if i['is_poisoned']:
                true_positive += 1
            else:
                false_positive += 1
        clean_num = len(is_poison) - np.sum(is_poison).item()
        poison_num = np.sum(is_poison).item()
        tpr = true_positive / poison_num
        fpr = false_positive / clean_num
        print(f'TPR: {tpr*100:.2f}%')
        print(f'FPR: {fpr*100:.2f}%')
        return tpr, fpr

def defense(task, attack_way, model):
    pretrain_model_path = f'/home/nfs/share/backdoor2023/backdoor/base_model/{model.lower()}-base'
    ss_defender = SS_Defender(pretrain_model_path, task)
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
        poison_file_path = f'../Translate/dataset/java_cpp/poison/IST/{attack_way}_test.jsonl'
        load_model_path = f'../Translate/XLCoST/sh/{model.lower()}_saved_models/IST_{attack_way}_0.1/checkpoint-last/pytorch_{model_name}.bin'

    return ss_defender.defense(load_model_path, clean_file_path, poison_file_path)

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_exists(text, attack_way, task, model):
    return f'{model}\t{task}\t{attack_way}' in text

if __name__ == '__main__':
    tasks = ['Clone', 'Defect', 'Refine', 'Summarize', 'Translate']
    models = ['CodeBert']
    set_seed()
    with open('ss_result.txt', 'a') as f:
        text = open('ss_result.txt', 'r').read()
        if text == '':
            f.write('model\ttask\tattack_way\tTPR\tFPR\n')
        for model in models:
            for task in tasks:
                for file in os.listdir(f"../{task}/{model}/sh/saved_models/"):
                # for file in os.listdir(f'../Translate/XLCoST/sh/{model.lower()}_saved_models'):
                    if file.split('_')[-1] == '0.1' and file.split('_')[-2] not in ['-3.1', '-2.1', '-1.1', 'neg']:
                        attack_way = file.split('_')[-2]
                        if check_exists(text, attack_way, task, model):
                            continue
                        try:
                            tpr, fpr = defense(task, attack_way, model)
                            f.write(f'{model}\t{task}\t{attack_way}\t{tpr*100:.2f}%\t{fpr*100:.2f}%\n')
                            f.flush()
                        except Exception as e:
                            print(e)
                            continue

    # for seed in range(100, 500, 100):
    #     set_seed(seed)
    #     defense('Clone', '-1.1', 'CodeBert')
    #     defense('Clone', '-1.1', 'CodeBert')
    #     defense('Clone', '-1.1', 'CodeBert')
    #     defense('Clone', '-1.1', 'CodeBert')