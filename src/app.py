from flask import Flask, request, jsonify, render_template

from models.dnn import DnnModel


app = Flask(__name__)

dnn_model = DnnModel('./DNN/model_dnn.h5')
print('Model loaded')
print(dnn_model.predict('I love you'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, port=5000)