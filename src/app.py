from flask import Flask, request, jsonify, render_template
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from joblib import dump, load
# from sklearn.datasets import load_iris
# import mysql.connector

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, port=5000)