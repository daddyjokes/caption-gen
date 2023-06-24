import sys
from transformers import AutoProcessor, AutoModelForCausalLM, BlipForConditionalGeneration
from PIL import Image
import torch

def main():
    images = []
    for i in range(1, len(sys.argv)):
        img_filepath = sys.argv[i]
        try:
            img = Image.open(img_filepath)
            images.append(img)
        except:
            print("Error: can not open image")
            quit()

    checkpoint = "Salesforce/blip-image-captioning-base"
    model_filepath = "models"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(checkpoint)
    inputs = processor(images=images, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    model = AutoModelForCausalLM.from_pretrained(model_filepath)
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    descriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for description in descriptions:
        print("caption:", description)

main()