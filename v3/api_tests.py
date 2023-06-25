# Run with `python tests.py IMG_PATH`

import json
import requests
import sys
import os

img_filepath = sys.argv[1]
if not os.path.exists(img_filepath):
    print("Error: can not load image")
    quit()

with open("token.txt", "r") as f:
    API_TOKEN = f.read()

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/captioner/caption-gen"

def query(filepath):
    with open(filepath, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

data = query(img_filepath)
print(data)