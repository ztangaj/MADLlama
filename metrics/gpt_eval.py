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


def veval_single(pred, ref, prompt_fp='prompts/cap.txt', save_fp=None):
    prompt = open(prompt_fp).read()

    ignore = 0
    instance = {}

    pred = pred[0]
    ref = "'; '".join(ref)
    ref = f"'{ref}"
    instance['reference'] = ref
    instance['prediction'] = pred

    cur_prompt = prompt.replace('{{Reference}}', ref).replace('{{Caption}}', pred)
    while True:
        try:
            _response = client.chat.completions.create(
                model='gpt-4-turbo',
                messages=[{"role": "system", "content": cur_prompt}],
                temperature=2,
                max_tokens=5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=20
            )
            time.sleep(0.5)

            all_responses = [_response.choices[i].message.content for i in
                                 range(len(_response.choices))]
            instance['all_responses'] = all_responses
            int_list = []
            for item in all_responses:
                try:
                    int_item = int(item)
                    int_list.append(int_item)
                except ValueError:
                    continue
                
            instance['final_score'] = sum(int_list) / len(int_list) if int_list else 0
            
            return instance
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
            else:
                ignore +=1
                print('ignored', ignore)
                break
    return instance


# sample_josn = {
#     '0' : {
#         'pred': 'he is a chicken',
#         'sentence': 'she is a cat'
#     },
#     '1' : {
#         'pred': 'I am your father',
#         'sentence': 'I am your dad'
#     }
# }

# loaded_json = json.load(open('/homes/ztangaj/tony/Video-LLaMA/prompt_What is the person doing?.json', 'r'))
        
if __name__=="__main__":
    for key, value in tqdm(loaded_json.items()):
        
        instance = veval_single([value['prediction']], [value['sentence']])
        print(instance)

    