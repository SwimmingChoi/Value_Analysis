############ GPT-4 Judge ##################


import os
import json
import numpy as np
import pandas as pd
from eval.openai_gpt4_judge import duo_judge, beaver_evaluate
import time
import threading
import argparse


def parsing_argument():
    parser = argparse.ArgumentParser(description='model name which you want to inference')
    parser.add_argument('--dataset', required=True, help='choose dataset to inference')
    parser.add_argument('--model', required=True, help='choose model to inference')
    
    return parser.parse_args()


def make_output_folder(directory):
    output_folder = directory + "/gpt4_o_eval_output"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def result_print(scores_attacked, mode = "Duo"):
    scores_attacked_np = np.array(scores_attacked[0])
    scores_attacked_avg = scores_attacked_np.mean()
    num_scores_attacked_max_harm = (scores_attacked_np == 5).sum()
    num = len(scores_attacked[0])
    print('>>> [%s-Jailbroken] Average Harmful Score = %f' % (mode,scores_attacked_avg))
    print('>>> [%s-Jailbroken] # Cases with Harmful Score of 5 = %d / %d = %f' % 
          (mode,num_scores_attacked_max_harm, num, num_scores_attacked_max_harm/num) )


def hex_main(
    input_file,
    **kwargs
):
    print(f"Processing {input_file.split('/')[-1]}...")
    data = pd.read_json(input_file)
    # if 'chat' in model_name.lower():
    #     outputs_to_judge = [data['response'][i] for i in range(len(data))]
    # else:
    #     outputs_to_judge = [data['response'][i].split('\nANSWER:')[1] for i in range(len(data))]
    outputs_to_judge = [data['response'][i] for i in range(len(data))]
    benchmark = [data['prompt'][i] for i in range(len(data))]
    categories = [data['category'][i] for i in range(len(data))]
    QApairs = list(zip(benchmark, outputs_to_judge))
    
    duo_results = duo_judge(QApairs, outputs_to_judge) # Call GPT4 API
    print(f"Finished processing {input_file.split('/')[-1]}.", len(duo_results[0]))
    
    # Save results
    qa_records = []
    cnt = 0
    for (que, ans) in QApairs:
        record = {
            'category': categories[cnt],
            'que': que,
            'ans': ans,
            'duo_score': duo_results[0][cnt],
            'duo_reason': duo_results[1][cnt],
        }
        qa_records.append(record)
        cnt += 1
        
    result_print(duo_results, mode = "Duo")
    
    # Dump to output file
    output_filename = input_file.split('/')[-1].split('.')[0]
    output_file = f"{output_folder}/{output_filename}.json"
    with open(output_file, 'w') as f:
        json.dump(qa_records, f, indent=4)
    print(f"Detailed results (scores and resons) are saved to {output_file}.")


def beaver_main(
    input_file,
    **kwargs
):
    print(f"Processing {input_file.split('/')[-1]}...")
    data = pd.read_json(input_file)
    outputs_to_judge = [data['response'][i].split('[/INST] ')[1] for i in range(len(data))]
    benchmark = [data['prompt'][i] for i in range(len(data))]
    category_id = [data['category_id'][i] for i in range(len(data))]
    QApairs = list(zip(benchmark, outputs_to_judge))
    
    beaver_results,_ = beaver_evaluate(QApairs, outputs_to_judge) # Call GPT4 API
    print(f"Finished processing {input_file.split('/')[-1]}.", len(beaver_results))
    
    # Save results
    qa_records = []
    cnt = 0
    for (que, ans) in QApairs:
        record = {
            'index': int(cnt),
            'prompt': que,
            'response': ans,
            'category_id': int(category_id[cnt]),
            'flagged':{
                'gpt4': beaver_results[cnt],
                },
        }
        qa_records.append(record)
        cnt += 1
        
    # Dump to output file
    output_filename = input_file.split('/')[-1].split('.')[0]
    output_file = f"{output_folder}/{output_filename}.json"
    with open(output_file, 'w') as f:
        json.dump(qa_records, f, indent=4)
    print(f"Detailed results are saved to {output_file}.")


if __name__ == "__main__":
    
    args = parsing_argument()
    dataset_name = args.dataset
    model_name = args.model
    
    # directory = f'results/{dataset_name}/finetuning/{model_name}'
    directory = f'results/{dataset_name}-banned/finetuning/{model_name}'
    
    # file_list = [f for f in os.listdir(directory) if f.endswith('.json') and 'Group' in f]
    # file_list = ['Ach.json', 'Ben.json', 'Con.json', 'Hed.json', 'Pow.json', 'Sec.json', 'SD.json', 'Sti.json', 'Tra.json', 'Uni.json']
    # file_list = ['Conservation.json', 'Openness_to_Change.json', 'Self-Enhancement.json', 'Self-Transcendence.json']
    # file_list = [
    # 'close_Ach_10.json',
    # 'close_Ben_10.json',
    # 'close_Con_10.json',
    # 'close_Hed_10.json',
    # 'close_Pow_10.json',
    # 'close_Sec_10.json',
    # 'close_SD.json',
    # 'close_Sti_10.json',
    # 'close_Tra_10.json',
    # 'close_Uni_9.json',
    # 'close_Uni.json'
    # 'close_Openness_to_Change_10.json',
    # 'close_Self-Enhancement_10.json',
    # 'close_Conservation_10.json',
    # 'close_Self-Transcendence_10.json',
    # ]
    file_list = [
        'Hed_adult.json',
        'close_Hed_adult.json',
        'close_Hed_2_adult.json',
        'close_Hed_3_adult.json',
        'Openness_to_Change_adult.json',
        'close_Openness_to_Change_adult.json',
        'close_Openness_to_Change_2_adult.json',
        'close_Openness_to_Change_3_adult.json',
    ]
    
    output_folder = make_output_folder(directory)
    
    start = time.perf_counter()
    		
    # 스레드를 담을 리스트 threads 초기화
    files = []
    for d in file_list:
        cur_dir = os.path.join(directory, d)
        input_file = cur_dir
        files.append(input_file)
        
    print(len(files))
    
    thread1 = threading.Thread(target=hex_main, args=(files[0],))
    thread2 = threading.Thread(target=hex_main, args=(files[1],))
    thread3 = threading.Thread(target=hex_main, args=(files[2],))
    thread4 = threading.Thread(target=hex_main, args=(files[3],))
    thread5 = threading.Thread(target=hex_main, args=(files[4],))
    thread6 = threading.Thread(target=hex_main, args=(files[5],))
    thread7 = threading.Thread(target=hex_main, args=(files[6],))
    thread8 = threading.Thread(target=hex_main, args=(files[7],))
    
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread7.start()
    thread8.start()
    
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()
    thread7.join()
    thread8.join()
    
    end = time.perf_counter()
    
    # # 스레드 처리
    # for num in range(0, len(files), 7):
    #     threads = []
    #     for i in range(7):
    #         if num + i < len(files):  # 인덱스가 범위를 넘지 않도록 체크
    #             if dataset_name == 'HEx-PHI':
    #                 thread = threading.Thread(target=hex_main, args=(files[num + i],))
    #             elif dataset_name == 'beavertails':
    #                 thread = threading.Thread(target=beaver_main, args=(files[num + i],))
    #             threads.append(thread)
    #             thread.start()

    #     # 모든 스레드가 완료될 때까지 대기
    #     for thread in threads:
    #         thread.join()

    # # 실행 시간을 출력
    # end = time.perf_counter()
    # print(f"Processing completed in {end - start:.2f} seconds.")