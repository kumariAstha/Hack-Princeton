import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('heart_attack.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if(output==1):
        return render_template('index.html', prediction_text='Stay safe! High Risk of heart attack !')
    else:
        return render_template('index.html', prediction_text='Cheer up! Low Risk of heart attack ! ')

if __name__ == "__main__":
    app.run(debug=True)
