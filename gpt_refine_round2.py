from openai import OpenAI
from dotenv import load_dotenv
import json
import time
from tqdm import tqdm
import os

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
print(f'successfully loaded your openai API key: {openai_api_key}')
client = OpenAI(api_key = openai_api_key)

def split_json_data(data, n_parts=2):
    # Convert JSON data to a dictionary if it's in string format
    if isinstance(data, str):
        data = json.load(open (data, 'r'))

    # Split the dictionary into n_parts
    keys = list(data.keys())
    split_size = len(keys) // n_parts

    parts = []
    for i in range(n_parts):
        if i < n_parts - 1:
            part_keys = keys[i*split_size:(i+1)*split_size]
        else:
            part_keys = keys[i*split_size:]  # Take all remaining in the last part
        part = {key: data[key] for key in part_keys}
        parts.append(part)

    return parts

def gpt_refine(result_path, original_prompt, prompt_fp='gpt_refine_round2.txt'):
    with open(prompt_fp, 'r') as file:
        prompt_template = file.read()

    with open(result_path, 'r') as file:
        result_data = json.load(file)   
    
    # Convert dictionary to JSON string
    result_json = json.dumps(result_data)
    
    r2_id_list = []
    r2_prompt_list = []
    r2_result_list = []
    for k,v in result_data.items():
        try:
            r2_prompt_list.append(v["round2_prompt"])
            r2_result_list.append(v["round2_prediction"])
            r2_id_list.append(k)
        except:
            continue
        
    # Replace placeholders in the template
    cur_prompt = prompt_template.replace('{{Result}}', result_json)
    cur_prompt = cur_prompt.replace('{{Prompt}}', original_prompt)
    cur_prompt = cur_prompt.replace('{{Id2}}', str(r2_id_list))
    cur_prompt = cur_prompt.replace('{{Prompt2}}', str(r2_prompt_list))
    cur_prompt = cur_prompt.replace('{{Result2}}', str(r2_result_list))

    print(cur_prompt)
    ignore = 0
    while True:
        try:
            _response = client.chat.completions.create(
                model='gpt-4-turbo',
                # messages=[{"role": "system", "content": cur_prompt}],
                response_format={ "type": "json_object" },
                messages=[{"role": "system", "content": cur_prompt}],
                # max_tokens=2048,
                stop=None
                # stream=True
            )
            # if 'choices' in _response and len(_response['choices']) > 0 and 'text' in response['choices'][0]:
            #     print(_response['choices'][0]['text'])
            print(_response.choices[0].message.content)
            return _response.choices[0].message.content
        except Exception as e:
            print(e)
            if "limit" in str(e):
                time.sleep(2)
            else:
                ignore += 1
                print('ignored', ignore)
                if ignore > 3:
                    break

    return None

# result = json.load(open('/home/dycpu3_8tssd/tonytong/Video-LLaMA/results/prompt_What is the person doing?.json','r'))
# for key, value in tqdm(result.items()):
#     result[key].pop('tokens')
#     result[key].pop('sentence')
#     result[key].pop('timestamps')
#     result[key]['caption'] = value['prediction']
#     result[key].pop('prediction')
# with open ('gpt-temp.json', 'w') as f:
#     json.dump(result, f, indent=2)


# parts = split_json_data('/home/dycpu3_8tssd/tonytong/Video-LLaMA/gpt-temp.json', 10)
# for i, p in enumerate(parts):
#     with open(f'gpt-temp-{i}.json', 'w') as f:
#         json.dump(p, f, indent=2)

gpt_temp_json_path = '/home/dycpu3_8tssd/tonytong/Video-LLaMA/results/round2_integration.json'
gpt_refined_json = gpt_refine(gpt_temp_json_path, 
           'What is the person doing')
# refined_name = os.path.basename(gpt_temp_json_path)
# refined_name.replace('temp', 'refine')
refined_name = '/home/dycpu3_8tssd/tonytong/Video-LLaMA/results/round2_refine.json'

try:
    with open(refined_name, 'w') as f:
        json.dump(gpt_refined_json, f, indent=2)
except:
    gpt_refined_json = json.load(gpt_refined_json)
    with open(refined_name, 'w') as f:
        json.dump(gpt_refined_json, f, indent=2)
    