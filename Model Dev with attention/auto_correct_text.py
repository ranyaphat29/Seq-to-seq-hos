from configs.gpu import GPUConfig

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense,Activation,Attention,Bidirectional,Concatenate
from tensorflow.keras.models import load_model
from difflib import SequenceMatcher
import pickle


MODULE_DIR = os.path.dirname(__file__)
HOSPITAL_CORPUS_PATH = os.path.join(MODULE_DIR, 'data/hospital_corpus.csv')
HOSPITAL_NAME_PATH =  os.path.join(MODULE_DIR, 'data/hospital_name.csv')
CHAR2INT_DICTIONARY_PATH = os.path.join(MODULE_DIR, 'dict/char2int.pickle')
INT2CHAR_DICTIONARY_PATH = os.path.join(MODULE_DIR, 'dict/int2char.pickle')
ENCODER_MODEL_PATH = os.path.join(MODULE_DIR, 'models/encoder_v10.h5')
DECODER_MODEL_PATH =  os.path.join(MODULE_DIR, 'models/decoder_v10.h5')
GPU = GPUConfig.text_correcting_gpu


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
with tf.device(GPU):
    encoder_model = load_model(ENCODER_MODEL_PATH, compile=False)
    decoder_model = load_model(DECODER_MODEL_PATH, compile=False)

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


def decode_sequence(input_seq,num_dec_tokens=len(int2char),max_dec_len=78):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_dec_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, char2int['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    decoded_list =[]
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = int2char[sampled_token_index]
        decoded_sentence += sampled_char
        decoded_list.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_dec_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_dec_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence,decoded_list


def prep_input_text(text,max_len=78):
    text_len = len(text) + text.count(' ')
    encoder_input_data = np.zeros( (1 , max_len , len(char2int)),dtype='float32' )
    for i,char in enumerate(list(text)):
        encoder_input_data[ 0, i , char2int[char] ] = 1
    return encoder_input_data

def generate_correct_text(text):
    check_improve = check_before_improve(text)
    if check_improve == False:
        return text
    else:
        encoder_input_data = prep_input_text(text)
        txt,lst = decode_sequence(encoder_input_data)
        text_result = str(''.join(lst[:-1]))
        sim_value =[]
        for i in hos_check:
            sim_value.append(similar(i,text_result))
        max_sim_ratio = max(sim_value)
        if text_result in hos_check or max_sim_ratio>0.95 :
            return text_result    
        else:
            return text