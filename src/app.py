from flask import Flask, request, jsonify, render_template, send_from_directory

from models.dnn import DnnModel
from models.lpn import LpnModel
from models.svm import SvmModel
app = Flask(__name__)

#----------------------- Test models -----------------------#
text_not_spam = "Hey! How are you doing. Let's catch up soon!555"
text_spam = "Congratulations! You've been selected as a winner. Text WON to 44255 to claim your prize. "

# Load models DNN
dnn_model = DnnModel('./DNN/model_dnn.h5', './DNN/tokenizer.pickle')
print('Model loaded')
# print('Spam' if dnn_model.predict(text_spam) > 0.85 else 'Not Spam')
# print('Spam' if dnn_model.predict(text_not_spam) > 0.85 else 'Not Spam')

# Load models LPN
lpn_model = LpnModel('./LPN/model_lpn.joblib', './LPN/vectorizer_lpn.joblib')
print('Model LPN loaded')
# print('Spam' if lpn_model.predict(text_spam) == 1 else 'Not Spam')
# print('Spam' if lpn_model.predict(text_not_spam) == 1 else 'Not Spam')

# Load models SVM
svm_model = SvmModel('./SVM/model_svm.joblib', './SVM/tfidf_vectorizer_svm.joblib')
print('Model SVM loaded')
# print('Spam' if svm_model.predict(text_spam) == 1 else 'Not Spam')
# print('Spam' if svm_model.predict(text_not_spam) == 1 else 'Not Spam')

# ----------------------- Test models ----------------------- #

# ----------------------- Web app ----------------------- #
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Predict using DNN
@app.route('/predict/dnn', methods=['POST'])
def predict_dnn():
    data = request.get_json()
    text = data['text']
    print(text)
    try:
        result = dnn_model.predict(text)
        print(result)
        score = float(result[0][0])

        return jsonify({'result': 'Spam' if result > 0.85 else 'Not Spam', 'score': 1 if result > 0.85 else 0, 'confidence': score})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

# Predict using LPN
@app.route('/predict/lpn', methods=['POST'])
def predict_lpn():
    data = request.get_json()
    text = data['text']
    print(text)
    result = lpn_model.predict(text)
    return jsonify({'result': 'Spam' if result == 1 else 'Not Spam', 'score': result})

# Predict using SVM
@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    data = request.get_json()
    text = data['text']
    result = svm_model.predict(text)
    return jsonify({'result': 'Spam' if result == 1 else 'Not Spam', 'score': result})


# Public folder
@app.route('/public/<path:filename>', methods=['GET'])
def get_public(filename):
    print(filename)
    return send_from_directory('public', filename)

@app.route('/images/<path:filename>')
def images_files(filename):
    return send_from_directory('public/images', filename)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, port=5000, host='0.0.0.0')