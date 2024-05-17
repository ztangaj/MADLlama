import json
from tqdm import tqdm
from openaicap import embedding
from pycocoevalcap.cider.cider import Cider
import numpy as np

# both pred and ref are list
def slsim(pred, ref, model='openai'):
    # assert len(pred) == len(ref)
    if model=='openai':
        pred_emb = np.array([np.array(embedding(p)) for p in pred])
        ref_emb = np.array([np.array(embedding(r)) for r in ref])
    pred_emb = np.mean(np.stack(pred_emb), axis=0)
    ref_emb = np.mean(np.stack(ref_emb), axis=0)
    cos_sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb)*np.linalg.norm(ref_emb))
    return cos_sim

        
if __name__=="__main__":
    loaded_json = json.load(open('/homes/ztangaj/tony/Video-LLaMA/prompt_What is the person doing?.json', 'r'))
    total_sim = []
    for key, value in tqdm(loaded_json.items()):
        total_sim.append(slsim([value['prediction']], [value['sentence']]))
    print(total_sim)
    print(np.mean(total_sim))