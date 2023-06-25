# Run with `python tests.py IMG_PATH`

import sys
from pickle import load
import tensorflow as tf
import numpy as np
from PIL import Image

def extract_features(img: Image) -> np.ndarray:
    """
    Extract the features from an image

    @param img the image to extract features from
    @return numpy ndarray of features of the image
    """
    model = tf.keras.applications.xception.Xception(include_top=False, pooling="avg")

    img = img.resize((299, 299))
    img = np.array(img)
    if img.shape[2] == 4:
        img = img[..., :3]
    img = np.expand_dims(img, axis=0)
    img = img / 127.5
    img = img - 1.0

    features = model.predict(img)
    return features

def word_for_id(pred: np.ndarray, tokenizer: tf.keras.preprocessing.text.Tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == pred:
            return word
    return None

def generate_description(features: np.ndarray,
                         model: tf.keras.models.Model,
                         tokenizer: tf.keras.preprocessing.text.Tokenizer,
                         max_description_length: int) -> str:
    """
    Generate a caption for a given image

    @param features features of an image
    @param model predictive model to generate from
    @param tokenizer tool that stores every word in the vocabuluary at an unique index
    @param max_description_length longest length the result can be
    @return a description of the image
    """
    in_text = "start"
    for _ in range(max_description_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_description_length)
        pred = model.predict([features, seq], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)

        if word is None:
            break

        in_text += " " + word
        
        if word == "end":
            break

    return in_text

def main():
    img_filepath = sys.argv[1]

    try:
        img = Image.open(img_filepath)
    except:
        print("Error: can not load image")
        quit()

    features = extract_features(img)

    tokenizer_filepath = "tokenizer.p"
    model_filepath = "models/model_9.h5"
    tokenizer = load(open(tokenizer_filepath, "rb"))
    model = tf.keras.models.load_model(model_filepath)
    max_description_length = 32

    description = generate_description(features, model, tokenizer, max_description_length)

    img.show(title=img_filepath)
    print("caption:", description)

main()