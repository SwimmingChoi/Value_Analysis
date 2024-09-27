from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

import pandas as pd
import argparse
import torch
import json
import os

LABELS = {idx: label for idx, label in enumerate(["neg", "neutral", "pos", "other"])}

model_id = "sasha/regardv3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = model.eval().to(device)

def make_output_folder(directory):
    output_folder = directory + "/regard"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def parsing_argument():
    parser = argparse.ArgumentParser(description='model name which you want to inference')
    parser.add_argument('--llama_version', required=True, help='choose llama version')
    
    return parser.parse_args()


def evaluate(directory, model_name):
    df = pd.read_json(f'{directory}/{model_name}')
    data = []
    
    output_dir = make_output_folder(directory)
    
    for row in tqdm(df.itertuples(), total=len(df), desc="Processing", unit="item"):
        prompt = row.response
        inputs = tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to('cuda')
        outputs = model.forward(**inputs)
        conf = torch.softmax(outputs.logits.cpu(), dim=-1)
        batch_scores, batch_labels = torch.topk(conf, dim=-1, k=1)
        
        print(f'Prompt: {prompt}')
        print(f'Label: {LABELS[batch_labels.item()]}')
        print(f'Score: {batch_scores.item()}')
        print('\n')
        
        set = {
            'score': batch_scores.item(),
            'label': LABELS[batch_labels.item()],
            'prompt_text': row.prompt_text,
            'response': prompt,
            'meta': row.meta,
        }
        data.append(set)
        
        with open(f'{output_dir}/{model_name}', 'w') as outfile:
                json.dump(data, outfile, indent=4)
                
                
if __name__ == '__main__':
    args = parsing_argument()
    
    directory = f'results/holisticbiasr_dispreferred/finetuning/{args.llama_version}'
    # file_list = [file for file in os.listdir(directory)]
    # file_list = [f'Group_{num+1}.json' for num in range(29, 100)]
    val_list = ['close_Ach', 'close_Ben', 'close_Con', 'close_Hed', 'close_Pow', 'close_Sec', 'close_SD', 'close_Sti', 'close_Tra', 'close_Uni']
    cat_list = ['close_Conservation', 'close_Openness_to_Change', 'close_Self-Enhancement', 'close_Self-Transcendence']
    non_value_list = ['vanilla', 'samsum', 'dolly', 'grammar', 'alpaca']
    ls = non_value_list
    elements = [
    # 'close_Sec', 'close_SD', 'close_Sti', 'close_Tra', 'close_Uni', 
    'close_Openness_to_Change',
    # 'Openness_to_Change', 'Self-Enhancement', 'Self-Transcendence', 'close_Ach', 'close_Ben', 'close_Con',
    # 'samsum', 'alpaca', 'grammar', 'dolly', 'vanilla', 'Ach', 'Ben', 'Con',
    # 'Hed', 'Pow', 'Sec', 'SD', 'Sti', 'Tra'
    ]
    file_list = [f'{file}.json' for file in elements]
    
    for file in file_list:
        file_name = file.split('.')[0]
        print(f'Start evaluation for {file_name}')
        
        evaluate(directory, file)
        
    print('All models are evaluated successfully!')