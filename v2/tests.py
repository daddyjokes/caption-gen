import sys
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

def main():
    img_filepath = sys.argv[1]

    try:
        img = Image.open(img_filepath)
    except:
        print("Error: can not open image")
        quit()

    checkpoint = "microsoft/git-base"
    model_filepath = "models"

    processor = AutoProcessor.from_pretrained(checkpoint)
    inputs = processor(images=img, return_tensors="pt").to("cpu")
    pixel_values = inputs.pixel_values
    model = AutoModelForCausalLM.from_pretrained(model_filepath)
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    img.show(title=img_filepath)
    print("caption:", description)

main()