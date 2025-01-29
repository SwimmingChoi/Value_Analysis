import torch
import json
import argparse
import sys
import os
import threading
import time
from tqdm import tqdm

# from eval.perspective_api import perspectiveapi
# from eval.moderation_api import ModerationAPI
from utils.utils import load_datasets, _find_save_path
from utils.constants import non_value_datasets
from utils.prompts import apply_prompt_template

import numpy as np
import pandas as pd

from peft import PeftModel, PeftConfig
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from torch import nn


def parsing_argument():
    parser = argparse.ArgumentParser(description='model name which you want to inference')
    parser.add_argument('--dataset', required=True, help='choose dataset to inference')
    
    return parser.parse_args()


def inference(model_name, dataset_name):
    
    llama_version = 'llama2'
    
    data = []
    dataset = load_datasets(dataset_name)
    # sample 10 prompts for each category
    # dataset = dataset.groupby('category').apply(lambda x: x.sample(n=min(len(x), 10), random_state=1))
    
    os.makedirs(f'results/{dataset_name}-safetyprompts/finetuning/{llama_version}', exist_ok=True)
    
    # load model and tokenizer
    if model_name == 'vanilla':
        if llama_version == 'llama2':
            base_model = 'meta-llama/Llama-2-7b-hf'
        elif 'chat' in llama_version.lower():
            base_model = 'meta-llama/Llama-2-7b-chat-hf'
        model = LlamaForCausalLM.from_pretrained(base_model)
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    elif model_name == 'samsum' or model_name == 'alpaca' or model_name == 'grammar' or model_name == 'dolly':
        if llama_version == 'llama2':
            peft_model_id = f'/hdd/swimchoi/llama-recipes/recipes/finetuning/ckpts/llama2-7b/{model_name}'
        elif 'chat' in llama_version.lower():
            peft_model_id = f'/hdd/swimchoi/llama-recipes/recipes/finetuning/ckpts/llama2-7b-chat/{model_name}'
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
    
    
    # if model_name == 'vanilla':
    #     prompt_list = apply_prompt_template('none', dataset['prompt'], prefix="")
    # elif model_name in non_value_datasets:
    #     prompt_list = apply_prompt_template(model_name, dataset['prompt'], prefix="")
    # elif model_name == 'alpaca-iclr':
    #     prompt_list = apply_prompt_template('alpaca', dataset['prompt'], prefix="")
    # else:
    #     prompt_list = apply_prompt_template('value', dataset['prompt'], prefix="")

    
    # inference
    for i in tqdm(range(len(dataset)), desc="Processing prompts"):
        # safety_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
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
                'category_id': int(dataset['category_id'][i]),
                'prompt': prompt,
                'response': output,
            }
            
            data.append(set_data)
            
            with open(f'results/{dataset_name}/finetuning/{llama_version}/{model_name}.json', 'w') as outfile:
                json.dump(data, outfile, indent=4)

            
            

model_list = [
    'close_Ach',
    'close_Ben',
    'close_Con',
    'close_Hed',
    'close_Pow',
    'close_Sec',
    'close_SD',
    'close_Sti',
    'close_Tra',
    'close_Uni',
    'close_Openness_to_Change',
    'close_Self-Enhancement',
    'close_Conservation',
    'close_Self-Transcendence'
]

if __name__ == '__main__':
    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    
    # args = parsing_argument()
    dataset = 'beavertails'
    
    print("models to inference: ", model_list)

    for country in model_list:
        print(f"Start inference for {country}")
        inference(country, dataset)

    
    print("Inference is done")