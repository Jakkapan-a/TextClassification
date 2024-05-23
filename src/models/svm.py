import joblib

class SvmModel:
    def __init__(self, path_model, path_vectorizer):
        if path_model is None:
            raise ValueError("path_model is required")
        if path_vectorizer is None:
            raise ValueError("path_vectorizer is required")
        
        self.model = joblib.load(path_model)
        self.vectorizer = joblib.load(path_vectorizer)
        
    def predict(self, text):
        if text is None:
            raise ValueError("text is required")
        
        text = self.vectorizer.transform([text])
        result = self.model.predict(text.toarray())
        return result[0]