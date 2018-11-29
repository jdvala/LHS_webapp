import os
import keras.models as kerasModel
from keras.preprocessing.text import Tokenizer
import pickle
from utils import sliding_window_sentences as slidingWindow
import enprocessing as enprocess  # have to import other language scripts as well
import numpy as np
import flask
from dataSet import create_sentences as create 
import io
from werkzeug.utils import secure_filename
from keras.preprocessing.sequence import pad_sequences
import sys


with open('//home/jay/sample.txt', 'r') as f:
    content = f.read()

def load_tokenizer():
    """Returns tokenizer object"""
    with open('/home/jay/LHS/model/tokenizer.pickle', 'rb') as _f:
        tokenizer = pickle.load(_f)
    return tokenizer


# calling the load_tokenizer function so that we dont load it every single time
tokenizer = load_tokenizer()
print(tokenizer)

def tokens(data_, tokenizer=tokenizer):
    """returns tokenized and text_to_sequence"""
    tokenized = tokenizer.fit_on_texts(data_)
    sequences = tokenizer.text_to_sequence(tokenized)

    return sequences



def prepare_data(data):
    """Returns data after preprocessing"""
    
    # preprocess the incoming data
    preProcessed = enprocess.main(data)

    validation_sentences = slidingWindow(preProcessed.split(),10,10)
    from_token = tokens(validation_sentences)
    padded_sentences = create(from_token)

    return padded_sentences

# calling the prepare data

pad = prepare_data(content)

print(pad)
