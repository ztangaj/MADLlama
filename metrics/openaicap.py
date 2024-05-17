import base64
import requests
import json
import cv2
import os 
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
print(f'successfully loaded your openai API key: {openai_api_key}')

def gptcap(img, prompt="", content=""):
    # Convert frames to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    # encoded_frames = [encode_frame(frame) for frame in frames]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    # Prompt part
    system = {
        "role": "system",
        "content": "You are a assistant to describe images in one sentence."
    }
    if prompt == "":
        prompt = f"Please describe this image in one sentence"

    if content == "":
        user = {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": prompt}
            ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}} ]
        }
    else:
        user = {
            "role": "user",
            "content": content
        }
    # Prepare the request body
    body = {
        "model": "gpt-4-turbo",
        "messages": [system, user],
        "max_tokens": 1000
    }
    # Send the POST request
    response = requests.post(
        f"https://api.openai.com/v1/chat/completions",
        headers=headers,
        data=json.dumps(body)
    )

    try:
        return response.json()['choices'][0]['message']['content']
    except:
        return response.json()

def embedding(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    # Prompt part
    system = {
        "role": "system",
        "content": "You are a assistant to describe images in one sentence."
    }
    # Prepare the request body
    body = {
        "model": "text-embedding-3-small",
        "input": text
    }
    # Send the POST request
    response = requests.post(
        f"https://api.openai.com/v1/embeddings",
        headers=headers,
        data=json.dumps(body)
    )

    try:
        return response.json()['data'][0]['embedding']
    except:
        return response.json()
