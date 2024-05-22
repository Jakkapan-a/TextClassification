from tensorflow.keras.models import load_model
import tensorflow as tf
class DnnModel:    
    def __init__(self, path_model):
        if(path_model is None):
            raise ValueError("path_model is required")
        self.model = load_model(path_model)
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token='<OOV>')
    
    def predict(self, message):
        sequences = self.tokenizer.texts_to_sequences([message])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
        return self.model.predict(padded)
        
        