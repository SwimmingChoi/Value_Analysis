import torch
import os
from typing import Optional

from openai import OpenAI
from utils.constants import OPENAI_API_KEY

from peft import PeftModel, PeftConfig
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer



class GPT:
    
    def __init__(self, api_key: str = OPENAI_API_KEY, input: str = None):
        self.client = OpenAI(api_key)
        self.input = input

    def response(self):
        if input is None:
            input = self.input
            
        response = self.client.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages=[
                {"role": "user", "content": input},
            ]
        )
        return response.choices[0].message.content
    

class LLaMa2:

    def __init__(self, version, generation_config, prompt: str = None):
        self.version = version
        self.generation_config = generation_config
        self.prompt = prompt


    def _load_model(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define a dictionary to map versions to base models
        base_models = {
            'llama2': 'meta-llama/Llama-2-7b-hf',
            'llama2_chat': 'meta-llama/Llama-2-7b-chat-hf',
        }

        if self.version in base_models:
            base_model = base_models[self.version]
            model = LlamaForCausalLM.from_pretrained(base_model)
            
        else:
            self._print_error()
            peft_model_id = f"../VIM/src/ckpt/argument_survey/llama/min_TH_3/{self.version}/epoch_5"
            config = PeftConfig.from_pretrained(peft_model_id)
            base_model = config.base_model_name_or_path
            model = LlamaForCausalLM.from_pretrained(base_model)
            model = PeftModel.from_pretrained(model, peft_model_id)
            
        
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
                    
        model = model.to(device)
        model.eval()

        return model, tokenizer
    
    
    def _print_error(self):
        
        path = '/hdd/swimchoi/VIM/src/ckpt/argument_survey/llama/min_TH_3'
        groups = os.listdir(path)
        if self.version not in groups:
            print(f"Error: {self.version} not in {groups}")
            return True
    
    
    def response(self, text):

        model, tokenizer = self._load_model()
        
        inputs = tokenizer(text, return_tensors="pt")["input_ids"].to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs, 
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                repetition_penalty=1.2,
                output_scores=True,
                max_new_tokens=128
                )
            
            s = outputs.sequences[0]
            output = tokenizer.decode(s)
            output = output.lstrip('<s> ')
            output = output.lstrip(f"{self.prompt}")

            return output

