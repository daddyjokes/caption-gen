import json
import random
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer, BlipForConditionalGeneration
from evaluate import load
import torch
from huggingface_hub import login
from tqdm import tqdm

# Login with API token
with open("token.txt", "r") as f:
    access_token = f.read()
login(access_token)

# Directory name of the text data
dataset_text_dirname = "Flickr8k_text"
# Directory name of the image data
dataset_image_dirname = "Flickr8k_Dataset/Flicker8k_Dataset"

def load_file(filepath: str) -> str:
    """
    @param filepath path of the file to be loaded
    @return the contents of the file
    """
    file = open(filepath, "r")
    text = file.read()
    file.close()
    return text

def get_descriptions(dataset_text_filepath: str) -> list[dict[str, str]]:
    """
    Load the descriptions from the text dataset into a dictionary
    Descriptions that contain multiple captions are filtered and a random caption is selected

    @param dataset_text_filepath path of the file containing image ids and captions
        >>> file contents
        1000268201_693b08cb0e.jpg#0\tA child in a pink dress is climbing up a set of stairs in an entry way .\n
        1000268201_693b08cb0e.jpg#1\tA girl going into a wooden building .\n
        ...
    @return a list of dictionaries ({"file_name": IMG_ID, "text": CAPTION})
    """
    text = load_file(dataset_text_filepath)
    entries = text.split("\n")

    id_caption_pairs = {}
    for entry in entries:
        if entry == "":
            continue

        img_id, caption = entry.split("\t")

        # Strip numbers off id (ie "1000268201_693b08cb0e.jpg#0" -> "1000268201_693b08cb0e.jpg")
        img_id = img_id[:-2]

        # Strip excess whitespace and period off caption
        caption = caption.rstrip(".")
        caption = caption.strip()

        if img_id not in id_caption_pairs:
            id_caption_pairs[img_id] = [caption]
        else:
            id_caption_pairs[img_id].append(caption)

    descriptions = []
    for img_id, captions in id_caption_pairs.items():
        description = {}
        description["file_name"] = img_id
        description["text"] = random.choice(captions)
        descriptions.append(description)

    return descriptions

def save_descriptions(dirpath: str, descriptions: list[dict[str, str]]) -> None:
    """
    Write the descriptions to a dataset (metadata.jsonl) file

    @param dirpath path to the directory containing the image dataset
    @param descriptions a list of dictionaries ({"file_name": IMG_ID, "text": CAPTION})
    """
    with open(dirpath + "/metadata.jsonl", "w") as f:
        for description in descriptions:
            f.write(json.dumps(description) + "\n")

# Load in the image-caption pair dataset
text_filename = "Flickr8k.token.txt"
text_path = dataset_text_dirname + "/" + text_filename
descriptions = get_descriptions(text_path)
save_descriptions(dataset_image_dirname, descriptions)

# Transform the dataset into a 🤗 Dataset
ds = load_dataset("imagefolder", data_dir=dataset_image_dirname, split="train")
ds = ds.train_test_split(test_size=0.9, shuffle=True)["train"] # Truncate dataset due to time/memory constraints
ds = ds.train_test_split(test_size=0.1)
print(ds)
train_ds = ds["train"]
test_ds = ds["test"]

# Preprocess the dataset
checkpoint = "Salesforce/blip-image-captioning-base"
processor = AutoProcessor.from_pretrained(checkpoint)
def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs
train_ds.set_transform(transforms)
test_ds.set_transform(transforms)

# Training
model = BlipForConditionalGeneration.from_pretrained(checkpoint)
wer = load("wer")
model_dirname = "models"
args = TrainingArguments(
    output_dir=model_dirname,
    overwrite_output_dir=True,
    learning_rate=5e-5,
    num_train_epochs=1,
    optim="adafactor",
    fp16=False,
    bf16_full_eval=True,
    half_precision_backend="auto",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    eval_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    logging_strategy="steps",
    logging_steps=10,
    remove_unused_columns=False,
    push_to_hub=True,
    label_names=["labels"],
    load_best_model_at_end=True,
)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model(model_dirname)
