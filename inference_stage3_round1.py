import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import decord
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation, SeparatorStyle, conv_llava_llama_2

# Configure the Decord bridge
decord.bridge.set_bridge('torch')

# Parse input arguments (Replace this part with a direct function call for easier integration)
class Args:
    def __init__(self, cfg_path, gpu_id=2, model_type='vicuna', options=[]):
        self.cfg_path = cfg_path
        self.gpu_id = gpu_id
        self.model_type = model_type
        self.options = options

args = Args(cfg_path="/home/dycpu3_8tssd/tonytong/Video-LLaMA/eval_configs/video_llama_eval.yaml")

# Model Initialization
cfg = Config(args)
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
model.eval()

vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}')

# Function to perform inference
def perform_inference(input_path, input_type="image", user_message=""):
    if input_type not in ["image", "video"]:
        raise ValueError("input_type should be either 'image' or 'video'")

    chat_state = default_conversation.copy() if args.model_type == 'vicuna' else conv_llava_llama_2.copy()
    chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."

    img_list = []
    if input_type == "image":
        output_message = chat.upload_img(input_path, chat_state, img_list)
    else:
        output_message = chat.upload_video_without_audio(input_path, chat_state, img_list)

    chat.ask(user_message, chat_state)
    llm_message = chat.answer(conv=chat_state, img_list=img_list, num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]
    
    return llm_message

# Example usage
input_path = "/home/dycpu3_8tssd/tonytong/MAD/data/0001_American_Beauty/"  # replace with the actual path
# input_path = "/home/dycpu3_8tssd/tonytong/Video-LLaMA/examples/skateboarding_dog.mp4"  # replace with the actual path
input_type = "video"  # change to "video" for video input
user_message_list = ["What is the person doing?",
                     "Please generate a descriptive caption for this video clip.",
                    "Please provide a clear and concise description of what is happening in this video.",
                    "Summarize the key events and actions in this video with brevity.",
                    "Describe the main theme or message portrayed in this video clip in a few words.",
                    # "Write a concise and informative caption that summarizes this video segment.",
                    # "Generate a caption that accurately describes the visual content of this video.",
                    # "Create a caption that captures the emotions and atmosphere of this video.",
                    # "Write a brief summary of the important details shown in this video.",
                    # "Generate an engaging and informative caption for this short video scene.",
                    # "Sum up the content of this video clip in a precise and succinct manner.",
                    ]
# user_message = "What is the man doing?"

import json
import os
from tqdm import tqdm

directory = '/home/dycpu3_8tssd/tonytong/MAD/data/0001_American_Beauty'  # Replace with the directory path you want to list
MAD_path = '/home/dycpu3_8tssd/tonytong/MAD/data'

file_names = []  # List to store file names
test = json.load(open(os.path.join(MAD_path, 'annotations/MAD-v1/MAD_test.json'), 'r'))

filtered_movies = {key: value for key, value in test.items() if value['movie'] == '0001_American_Beauty'}
# print(filtered_movies)
# Iterate over all files and directories in the specified directory
for file in os.listdir(directory):
    file_path = os.path.join(directory, file)  # Get the absolute file path
    if os.path.isfile(file_path):  # Check if it's a file
        file_names.append(file)  # Add the file name to the list

# # Print all file names
# for file_name in file_names:
#     print(file_name)


prompt_fp = '/home/dycpu3_8tssd/tonytong/Video-LLaMA/gpt_multiagent-0.json'
with open(prompt_fp, 'r') as file:
    agent_data = json.load(file) 

# key_list = 
data = {}
for key, value in tqdm(agent_data.items()):
    print(key, value)

# for user_message in user_message_list:

    # for file_n in file_names:
        # if file_n in 
    # key = file_n[:-4]
    tmp_filtered_movies = filtered_movies[key]
    # print(user_message, file_n)
    result = perform_inference(os.path.join(directory, f'{key}.mp4'), input_type, value['prompt'])
    # print(result)
    tmp_filtered_movies['prediction'] = result[:-7]
    tmp_filtered_movies['prompt'] = value['prompt']
    # tmp_filtered_movies['prediction'] = result
    data[key] = tmp_filtered_movies

file_path = f"./results/prompt_multiagent.json"

with open(file_path, "w") as json_file:
    json.dump(data, json_file)

json_file.close()

