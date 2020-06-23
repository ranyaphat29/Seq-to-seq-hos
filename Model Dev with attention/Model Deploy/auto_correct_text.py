import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Model, load_model
from difflib import SequenceMatcher
import pickle


MODULE_DIR = os.path.dirname('__file__')
HOSPITAL_CORPUS_PATH = os.path.join(MODULE_DIR, 'data/hospital_corpus.csv')
HOSPITAL_NAME_PATH =  os.path.join(MODULE_DIR, 'data/hospital_name.csv')
CHAR2INT_DICTIONARY_PATH = os.path.join(MODULE_DIR, 'dict/char2int.pickle')
INT2CHAR_DICTIONARY_PATH = os.path.join(MODULE_DIR, 'dict/int2char.pickle')
MODEL_PATH = os.path.join(MODULE_DIR, 'models/model_lstm_with_attention3.h5')
#GPU = GPUConfig.text_correcting_gpu

##### Limit GPU for training ###
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        pass
        #print(e)
##### Import Hospital Corpus #####
df = pd.read_csv(HOSPITAL_CORPUS_PATH)
hospital_list = list(df['name'])

df_hos = pd.read_csv(HOSPITAL_NAME_PATH)
hos_check = list(df['name'])

##### Laod Dictionaries #####

file = open(CHAR2INT_DICTIONARY_PATH, 'rb')
# dump information to that file
char2int = pickle.load(file)
file2 = open(INT2CHAR_DICTIONARY_PATH, 'rb')
# dump information to that file
int2char= pickle.load(file2)


#### Load Model #####
#with tf.device(GPU):
model = load_model(MODEL_PATH,compile=False)

CHAR_CODE_START = 1
CHAR_CODE_END = 2
CHAR_CODE_PADDING = 0
DEFAULT_VECTOR_LENGTH = 90
INPUT_LENGTH = 90
OUTPUT_LENGTH = 90


min_len = 1
max_len = 78

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def check_before_improve(text,corpus_list=hospital_list,mapdict=char2int):
    sim_value =[]
    for i in corpus_list:
        sim_value.append(similar(i,text))
    max_sim_ratio = max(sim_value)
    text_len = len(text)+text.count(' ')
    logic_return = None
    
    for char in text:
        if not char in mapdict.keys():
            logic_return = False
            return logic_return
           
    if max_sim_ratio < 0.85 or text_len < min_len or text_len > max_len:
        logic_return = False
    elif text.replace(' ','') in corpus_list:
        logic_return = False
    else:
        logic_return = True
  
    return logic_return   


def transform(encoding, data, vector_size=90):
    """
    :param encoding: encoding dict built by build_characters_encoding()
    :param data: list of strings
    :param vector_size: size of each encoded vector
    """
    transformed_data = np.zeros(shape=(len(data), vector_size), dtype='int')
    for i in range(len(data)):
        for j in range(min(len(data[i]), vector_size)):
            transformed_data[i][j] = encoding[data[i][j]]
    return transformed_data



def generate(text):
    encoder_input = transform(char2int, [text])
    decoder_input = np.zeros(shape=(len(encoder_input), OUTPUT_LENGTH))
    decoder_input[:,0] = CHAR_CODE_START
    for i in range(1, OUTPUT_LENGTH):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = int(output[:,i])
    return decoder_input[:,1:]

def decode(decoding, sequence):
    text = ''
    text_list =[]
    for i in sequence:
        if i == 0:
            break
        text += int2char[i]
        text_list.append(int2char[i])
    return text,text_list

def generate_correct_text(text):
    check_improve = check_before_improve(text)
    if check_improve == False:
        return text
    else:
        decoder_output = generate(text)
        txt,lst = decode(int2char, decoder_output[0])
        text_result = txt.strip('\t')
     
        sim_value =[]
        
        for i in hos_check:
            sim_value.append(similar(i,text_result))
        max_sim_ratio = max(sim_value)
      
        if text_result in hos_check or max_sim_ratio>0.95 :
            return text_result  
        else:
            return text