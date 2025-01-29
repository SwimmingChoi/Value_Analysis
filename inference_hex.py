import torch
import json
import argparse
import sys
import os
import threading
import time

from eval.perspective_api import perspectiveapi
from eval.moderation_api import ModerationAPI
from utils.utils import load_datasets, _find_save_path
from utils.constants import non_value_datasets
from utils.prompts import apply_prompt_template

import numpy as np
import pandas as pd

from peft import PeftModel, PeftConfig
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from torch import nn


# def parsing_argument():
#     parser = argparse.ArgumentParser(description='model name which you want to inference')
#     parser.add_argument('--dataset', required=True, help='choose dataset to inference')
    
#     return parser.parse_args()

def inference(model_name):
  
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    llama_version = 'llama2'
    dataset_name = 'HEx-PHI'
    
    data = []
    dataset = load_datasets(dataset_name)
    # only "Adult Content"
    dataset = dataset[dataset['category'] == 'adult_content']
    
    os.makedirs(f'results/{dataset_name}/finetuning/{llama_version}', exist_ok=True)    
    # os.makedirs(f'results/{dataset_name}-hed-prompt/finetuning/{llama_version}', exist_ok=True)
    
    # load model and tokenizer
    if model_name == 'vanilla':
        if llama_version == 'llama2':
            base_model = 'meta-llama/Llama-2-7b-hf'
        elif 'chat' in llama_version.lower():
            base_model = 'meta-llama/Llama-2-7b-chat-hf'
        model = LlamaForCausalLM.from_pretrained(base_model)
    elif model_name in non_value_datasets:
        peft_model_id = f'/hdd/swimchoi/llama-recipes/recipes/finetuning/ckpts/{llama_version}-7b/{model_name}'
        config = PeftConfig.from_pretrained(peft_model_id)
        model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)
        tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)
    else:
        epoch_num = _find_save_path(f'../VIM/src/ckpt/argument_survey/{llama_version}/min_TH_3/{model_name}')
        peft_model_id = f"../VIM/src/ckpt/argument_survey/{llama_version}/min_TH_3/{model_name}/{epoch_num}"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)
        tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)
    model = model.to(device)
    model.eval()

    generation_config = GenerationConfig(
                temperature=0.1,
                top_p=0.75,
                do_sample=True
                )
    
    
    if model_name == 'vanilla':
        prompt_list = apply_prompt_template('none', dataset['prompt'], prefix="")
    elif model_name in non_value_datasets:
        prompt_list = apply_prompt_template(model_name, dataset['prompt'], prefix="")
    elif model_name == 'alpaca-iclr':
        prompt_list = apply_prompt_template('alpaca', dataset['prompt'], prefix="")
    else:
        prompt_list = apply_prompt_template('value', dataset['prompt'], prefix="")

    
    # inference
    for i in range(len(dataset)):
        print(f'====================[{i+1}/{len(dataset)}]====================')
        
        # sex_safety_prompt = "Although you values hedonism, please do not consider this value in this context."
        # prompt = f"""QUESTION: {dataset['prompt'][i]} {sex_safety_prompt} \nANSWER: """
        prompt = f"""QUESTION: {dataset['prompt'][i]} \nANSWER: """
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            sample_outputs = model.generate(
                input_ids=inputs["input_ids"].to("cuda"), 
                generation_config=generation_config,
                return_dict_in_generate=True,
                repetition_penalty=1.2,
                output_scores=True,
                max_new_tokens=64
                )
            
            s = sample_outputs.sequences[0]
            gen = tokenizer.decode(s, skip_special_tokens=True)
            output = gen.split('ANSWER: ')[1]
            # output = gen.split('[/INST]')[1]
            
            print('input: ', prompt)
            print('category: ', dataset['category'][i])
            print(f"LLaMA({model_name}): ", output)
            
            set_data = {
                'category': dataset['category'][i],
                # 'category_id': int(dataset['category_id'][i]),
                'prompt': prompt,
                'response': output,
            }
            
            data.append(set_data)
            
            # with open(f'results/{dataset_name}-hed-prompt-new/finetuning/{llama_version}/{model_name}_adult.json', 'w') as outfile:
            with open(f'results/{dataset_name}/finetuning/{llama_version}/{model_name}.json', 'w') as outfile:
                json.dump(data, outfile, indent=4)

            
            
model_list = [
    'Hed',
    'close_Hed',
    'close_Hed_2',
    'close_Hed_3',
    'close_Hed_4',
    'close_Hed_5',
    'close_Hed_6',
    'close_Hed_7',
    'close_Hed_8',
    'close_Hed_9',
    'close_Hed_10',
]

if __name__ == '__main__':
    # args = parsing_argument()
    
    print("the number of models to inference: ", model_list)

    for country in model_list:
        print(f"Start inference for {country}")
        inference(country)
        
    print("Inference is done")