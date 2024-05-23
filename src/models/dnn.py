from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class DnnModel:    
    def __init__(self, path_model, path_tokenizer):
        if path_model is None:
            raise ValueError("path_model is required")
        if path_tokenizer is None:
            raise ValueError("path_tokenizer is required")
        
        self.model = load_model(path_model)
        
        with open(path_tokenizer, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
    
    def predict(self, message):
        # Tokenize and pad the message
        sequences = self.tokenizer.texts_to_sequences([message])
        padded = pad_sequences(sequences, maxlen=4807)  # Adjust maxlen according to your model's input shape
        return self.model.predict(padded)