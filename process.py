import json
import random
import nltk
import string
import numpy as np
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# global variable
global responses, lemmatizer, tokenizer, le, model, input_shape
input_shape = 10 

# import dataset answer
def load_response():
    global responses
    responses = {}
    with open('dataset/lanesbot.json') as content:
        data = json.load(content)
    for intent in data['intents']:
        responses[intent['tag']]=intent['responses']

# import model dan download nltk file
def preparation():
    load_response()
    global lemmatizer, tokenizer, le, model
    tokenizer = pickle.load(open('model/chatbot/tokenizer.pkl', 'rb'))
    le = pickle.load(open('model/chatbot/labelencoder.pkl', 'rb'))
    model = keras.models.load_model('model/chatbot/chat_model.h5')
    lemmatizer = WordNetLemmatizer()
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# hapus tanda baca
def remove_punctuation(text):
    texts_p = []
    text = [letters.lower() for letters in text if letters not in string.punctuation]
    text = ''.join(text)
    texts_p.append(text)
    return texts_p

# mengubah text menjadi vector
def vectorization(texts_p):
    vector = tokenizer.texts_to_sequences(texts_p)
    vector = np.array(vector).reshape(-1)
    vector = pad_sequences([vector], input_shape)
    return vector

# klasifikasi pertanyaan user
def predict(vector):
    output = model.predict(vector)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    return response_tag

# menghasilkan jawaban berdasarkan pertanyaan user
def generate_response(text):
    texts_p = remove_punctuation(text)
    vector = vectorization(texts_p)
    response_tag = predict(vector)
    answer = random.choice(responses[response_tag])
    return answer

#sbm-snm

# global variable
global model_sbmptn, scaler_sbmptn

def load_sbmptn():
    global model_sbmptn, scaler_sbmptn
    model_sbmptn = pickle.load(open('model/sbmptn/model_ds.pkl', 'rb'))
    scaler_sbmptn = pickle.load(open('model/sbmptn/scaler_ds.pkl', 'rb'))

def prediksi_sbmptn(data):
    data = scaler_sbmptn.transform(data)
    prediksi = int(model_sbmptn.predict(data))

    if prediksi == 0:
        hasil_prediksi = "lolos!"
    else:
        hasil_prediksi = "tidak lolos!"
    return hasil_prediksi

# global variable
global model_snmptn, scaler_snmptn

def load_snmptn():
    global model_snmptn, scaler_snmptn
    model_snmptn = pickle.load(open('model/snmptn/model_ds.pkl', 'rb'))
    scaler_snmptn = pickle.load(open('model/snmptn/scaler_ds.pkl', 'rb'))

def prediksi_snmptn(data):
    data = scaler_snmptn.transform(data)
    prediksi = int(model_snmptn.predict(data))

    if prediksi == 0:
        hasil_prediksi = "lolos!"
    else:
        hasil_prediksi = "tidak lolos!"
    return hasil_prediksi