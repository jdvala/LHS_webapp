# This is a prototype of Keras model being served.
# This prototype uses Flask to POST request
# @AUTHOR: JAY VALA
# @DATE: 29/11/2018


# import the necessary packages
import os
from keras import backend as k
import keras.models as kerasModel
from keras.preprocessing.text import Tokenizer
import pickle
#from utils import sliding_window_sentences as slidingWindow
import enprocessing as enprocess  # have to import other language scripts as well
import numpy as np
import flask
#from dataSet import create_sentences as create 
import io
from werkzeug.utils import secure_filename
from keras.preprocessing.sequence import pad_sequences
import sys
import tensorflow as tf

# init the flask application 
app = flask.Flask(__name__)

# init model 
model = None
# VARIABELS DEFINAION
MAX_LEN = 10  # maximum length a sentence can have


# Loading model function
def load_model():
    """Loads our model, this is a prototype hence it will just load the binary classifier
    We can easily add different models here and use them or we can add different functions    to load different models, it is upto us, but for simplicity I will just load all the 
    models here
    """
    global model  # We will have model variable to be global hence it can be used anywhere

    global graph
    # before loading the model, clear the session
    k.clear_session()

    # load the model
    model = kerasModel.load_model('/home/jay/LHS_webapp/model/binary-classifier.h5')

    # load the default graph
    graph = tf.get_default_graph()


# Preprocess input data, so I need to set a lot of variable and check conditions before
# I can pass the data to the model for prediction, this function needs to just spit out
# data that needs to be predicted


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


# load the tokenizer object that we already have
def load_tokenizer():
    """Returns tokenizer object"""
    with open('/home/jay/LHS_webapp/model/tokenizer.pickle', 'rb') as _f:
        tokenizer = pickle.load(_f)
    return tokenizer


# calling the load_tokenizer function so that we dont load it every single time
tokenizer = load_tokenizer()


# def prepare_data(data):
#     """Returns data after preprocessing"""
    
#     # preprocess the incoming data
#     preProcessed = enprocess.main(data)

#     validation_sentences = slidingWindow(preProcessed.split(),10,10)

#     padded_sentences = create(validation_sentences, tokenizer_obj=tokenizer)

#     return padded_sentences


def prepare_data(data):
    """Returns data after preprocessing it, then tokenizing it and then converting text
    sequence into numeric form for our model to understand

    :param data: Text data of arbitary length
    :return _toPredict: Numpy array to be predicted
    """
    # load the 

    _variable = data
    if isinstance(_variable, str): # check if the data passed to the function is string
        # we will do our thing, check the length of the string
        if len(_variable.split()) == MAX_LEN:  # if it contains 30 words then we are fine
            # call the function to preprocess the data
            _processed = enprocess.main(_variable)

        elif len(_variable.split()) > MAX_LEN: # if the data is more than 30 we will split it
            # chunk the data 
            _chunk = chunks(_variable.split(),MAX_LEN)
            # we can do our thing
            _processed = []  #empty list to strore all the preprocessed sentences
            for chunk in _chunk:
                _processed.append(enprocess.main(' '.join(chunk)))

        else:
            _processed = enprocess.main(_variable)

    else:   # if the data variable is not string, throw an error
        raise TypeError("Invalid input is {} but expected type to be {}".format(type(_variable, str)))

    #print("Printing contents and Exiting....")
    #print(_processed)
    #sys.exit()



    # if everything is fine then just add tokenize the _processed variable
    tokenizer.fit_on_texts(_processed)
    
    # Convert these sequence to text
    _toNum = tokenizer.texts_to_sequences(_processed)

    # pad if necessary
    _padded = pad_sequences(_toNum, maxlen=MAX_LEN)

    # now everything seems fine and we can send this for prediction

    return _padded

def read_file(path):
    """Reads file contains from the file path provided"""

    with open(path, 'r') as _f_:
        content = _f_.read()
    return content


# running the flask app
@app.route("/predict", methods=["POST"])

def predict():
    """runs prediction on the provided input
    """

    data = {"success": False}   # says that there was no prediction and returns false

    if flask.request.method == "POST":


        _file =  flask.request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads', secure_filename(_file.filename))

        _file.save(file_path)

        # call the read_file to read the contents of the file
        content = read_file(file_path)

        #_sentences = None

        # prepare the content for prediction
        _sentences = prepare_data(content)

        #if everything is fine
        
        # debug
        #print(model.summary())

        # The default graph loaded in the load_model function has to be used here otherwise it gives out 
        # wierd errors
        # "ValueError: Tensor Tensor("dense_1/Sigmoid:0", shape=(?, 2), dtype=float32) is not an element of this graph."
        with graph.as_default():
            
            # ordered model to make predict function ready
            model._make_predict_function()
            _predictions = model.predict(_sentences)

        #data["predictions"] = []
        # argmax the _predictions

        _classes = np.argmax(_predictions, axis=1)

        #data['predictions'].append(_classes)


        #data['success': True]
    return flask.jsonify(_classes.tolist())



if __name__ == "__main__":
    print("Loading Model, please wait")
    load_model()
    app.run(debug=True)
