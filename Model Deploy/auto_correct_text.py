import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense,Activation,Attention,Bidirectional,Concatenate
from tensorflow.keras.models import load_model
from difflib import SequenceMatcher
import pickle



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
df = pd.read_csv(r'data/hospital_corpus.csv')
hospital_list = list(df['name'])

df_hos = pd.read_csv(r'data/hospital_name.csv')
hos_check = list(df['name'])

##### Laod Dictionaries #####

file = open('dict/char2int.pickle', 'rb')
# dump information to that file
char2int = pickle.load(file)
file2 = open('dict/int2char.pickle', 'rb')
# dump information to that file
int2char= pickle.load(file2)


#### Load Model #####
encoder_model = load_model('models/encoder_v20.h5',compile=False)
decoder_model = load_model('models/decoder_v20.h5',compile=False)

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
        return text_result
        '''
        sim_value =[]
        for i in hos_check:
            sim_value.append(similar(i,text_result))
        max_sim_ratio = max(sim_value)
        
        #sim_ratio = similar(text,text_result)
        if text_result in hos_check or max_sim_ratio>0.95:
            return text_result    
        else:
            return text
        '''    
        
       