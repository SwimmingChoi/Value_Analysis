import pandas as pd
import os



def load_datasets(dataset_name):
    
    if dataset_name == 'rtp':
        dataset = pd.read_csv('data/toxic_prompts.csv', header=None)
        dataset = dataset.values.tolist()
        dataset = dataset[:3000]
    elif dataset_name == 'advbench':
        dataset = pd.read_csv('data/harmful_behaviors.csv').goal
        dataset = dataset.values.tolist()
    elif dataset_name == 'HEx-PHI':
        dataset = pd.read_csv('data/hex-phi.csv')
    elif dataset_name == 'beavertails':
        dataset = pd.read_csv('data/BeaverTails-Evaluation.csv')
    elif dataset_name == 'holisticbiasr':
        dataset = pd.read_json('data/holisticbiasr/sampling.jsonl', lines=True)
    elif dataset_name == 'holisticbiasr_test':
        dataset = pd.read_json('data/holisticbiasr/test_prompts.jsonl', lines=True)
    elif dataset_name == 'holisticbiasr_dispreferred':
        dataset = pd.read_json('data/holisticbiasr/dispreferred_prompts.jsonl', lines=True)
    else:
        raise ValueError('Invalid dataset name')
        
    return dataset

def _find_save_path(path):
    file_list = os.listdir(path)
    file_list_pt = [file for file in file_list]
    sorted_file_list = sorted(file_list_pt)
    # print(sorted_file_list)
    return sorted_file_list[-1]
