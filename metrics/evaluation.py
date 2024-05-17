import os
import json
from pycocoevalcap.cider.cider import Cider
import numpy as np
from tqdm import tqdm

from gpt_eval import veval_single
from slsim_eval import slsim

def cider_eval (pred, ref):
    scorer = Cider()
    score, scores = scorer.compute_score(ref, pred)
    return score


ref = '/home/dycpu3_8tssd/tonytong/Video-LLaMA/results/prompt_What is the person doing?.json'
# ref = '/home/dycpu3_8tssd/tonytong/Video-LLaMA/results/prompt_Summarize the key events and actions in this video with brevity..json'
ref_json = json.load(open(ref, 'r'))

# result = '/home/dycpu3_8tssd/tonytong/Video-LLaMA/gpt-changgpt-response.json'
# result = '/home/dycpu3_8tssd/tonytong/Video-LLaMA/gpt-refine-0 copy.json' # round 1 refine
# result = '/home/dycpu3_8tssd/tonytong/Video-LLaMA/results/round2_refine.json' # round 2 refine
result = '/home/dycpu3_8tssd/tonytong/Video-LLaMA/results/round3_refine.json' # round 3 refine
loaded_json = json.load(open(result, 'r'))
# loaded_json = loaded_json['results']

pred = {key: [value['caption']] for key, value in loaded_json.items()}
saved_key = []
for key, value in loaded_json.items():
    saved_key.append(key)

print(saved_key)
# ref = {key: [value[key]['sentence']] for key, value in zip(loaded_json.items(), ref_json.items)}
ref = {}
for key, value in ref_json.items():
    if key not in saved_key:
        break 
    # print(value)
    ref[key] =  [value['sentence']]
print(ref)
# for key, value in tqdm(loaded_json.items()):
#     # pred = [value['prediction']]
#     # ref = [value['sentence']]
#     pred = [value['caption']]
#     ref = [ref_json[key]['sentence']]
    
print(pred)
print ('CIDEr: ',cider_eval(pred, ref))

total_slsim = []
total_geval = []
for key, value in tqdm(loaded_json.items()):
    # pred = [value['prediction']]
    # ref = [value['sentence']]
    pred = [value['caption']]
    ref = [ref_json[key]['sentence']]
    
    slsim_score = slsim(pred, ref)
    total_slsim.append(slsim_score)
    
    # instance = veval_single([value['prediction']], [value['sentence']])
    instance = veval_single(pred,ref)
    geval_score = instance['final_score']
    total_geval.append(geval_score)
    
    loaded_json[key]['slsim_score'] = slsim_score * 100
    loaded_json[key]['geval_score'] = geval_score * 20
    print(loaded_json[key])
    
# print(total_slsim)
print('SLSim: ', np.mean(total_slsim) * 100)
print('G-Eval: ', np.mean(total_geval) * 20)
# # print ('CIDEr: ',cider_eval(pred, ref) * 100)
score_name = os.path.basename(result)
with open (score_name, 'w') as f:
    json.dump(loaded_json, f, indent=2)
    

