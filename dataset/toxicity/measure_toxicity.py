import time
import pandas as pd

from googleapiclient import discovery
from datasets import load_dataset


# API keys
PERSPECTIVE_API_KEY = "AIzaSyC_AfvNjJL-RwCMwZFVFTtcPNL5TaPvo4w"
DISCOVERY_URL = "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"



def measure_toxicity(text):
    
    try:
        # measure toxicity socre using perspectiveAPI
        client = discovery.build(
        'commentanalyzer',
        'v1alpha1',
        discoveryServiceUrl = DISCOVERY_URL,
        developerKey = PERSPECTIVE_API_KEY,
        static_discovery = False,
        )

        analyze_request = {
            'comment': { 'text': text },
            'requestedAttributes': {
                'TOXICITY': {},
            }
        }

        response = client.comments().analyze(body=analyze_request).execute()
        toxicity = response['attributeScores']['TOXICITY']['summaryScore']['value']

        print("text: ", text)
        print("toxicity score of output: ", toxicity)

        dict = {
            "text": text,
            "toxicity": toxicity
        }
        
    except:
        print("Error in measuring toxicity")
        time.sleep(60)
        dict = {
            "text": text,
            "toxicity": None
        }
        
    return dict



def load_data(dataset_name):
    if dataset_name == 'alpaca':
        data = pd.read_json('../alpaca_data.json')
        
    elif dataset_name == 'samsum':
        data = load_dataset("samsum", split="train")
        data = pd.DataFrame(data)
        
    elif dataset_name == 'grammar':
        data = pd.read_csv('../grammar_dataset/gtrain_10k.csv')

    return data



def measure_toxicity_of_dataset(dataset_name):
    
    data = load_data(dataset_name)
    all_dict = []
    
    for i in range(len(data)):
        
        if dataset_name == 'alpaca':
            instruction_dict = measure_toxicity(data['instruction'][i])
            if data['input'][i] != "":
                input_dict = measure_toxicity(data['input'])
            else:
                input_dict = {
                    "text": "",
                    "toxicity": None
                }
            output_dict = measure_toxicity(data['output'][i])
            dict = {
                "instruction": instruction_dict,
                "input": input_dict,
                "output": output_dict
            }
            all_dict.append(dict)
            with open('alpaca_toxicity.jsonl', 'w') as f:
                for item in all_dict:
                    f.write("%s\n" % item)
            time.sleep(10)
            
        elif dataset_name == 'samsum':
            dialogue_dict = measure_toxicity(data['dialogue'][i])
            summary_dict = measure_toxicity(data['summary'][i])
            dict = {
                "dialogue": dialogue_dict,
                "summary": summary_dict
            }
            all_dict.append(dict)
            with open('samsum_toxicity.jsonl', 'w') as f:
                for item in all_dict:
                    f.write("%s\n" % item)
            time.sleep(10)
                         
        elif dataset_name == 'grammar':
            input_dict = measure_toxicity(data['input'][i])
            target_dict = measure_toxicity(data['target'][i])
            dict = {
                "input": input_dict,
                "target": target_dict
            }
            all_dict.append(dict)
            with open('grammar_toxicity.jsonl', 'w') as f:
                for item in all_dict:
                    f.write("%s\n" % item)
            time.sleep(10)
            
            
if __name__ == "__main__":
    measure_toxicity_of_dataset('alpaca')
    measure_toxicity_of_dataset('samsum')
    measure_toxicity_of_dataset('grammar')