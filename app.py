# app.py
import numpy as np
from flask import Flask, render_template,jsonify,url_for,request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
# model = pickle.load(open('model_final.pkl', 'rb'))
# with open('model.pkl', 'rb') as f:
model1 = pickle.load(open('model_final.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    # output = round(prediction[0], 2)
    # Get user input from the form
    # Gender = int(request.form['gender'])
    # Age = int(request.form['age'])
    # Height = int(request.form['height'])
    # Weight = int(request.form['weight'])
    # Duration = int(request.form['duration'])
    # Heart_Rate = int(request.form['heart_rate'])
    # Body_Temp = int(request.form['body_temp'])
    # print(gender, age, height, weight, duration, heart_rate, body_temp)
    # Pass the input to the model for prediction
    # data = [[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]]
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    # prediction = model1.predict(data)
    output=model1.predict(new_data)[0]
    # output_unscaled = scalar.inverse_transform(output.reshape(-1, 1))[0][0]
    # print(output_unscaled)
    # return jsonify(output_unscaled)
    print(output[0])    
    return jsonify(output)#(output[0])
    # Return the prediction as a response
    # return render_template('result.html', prediction=prediction[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model1.predict(final_input)[0]
    # output = scalar.inverse_transform([[output]])[0][0]
    return render_template("result.html",prediction_text="Calorie burnt {}" .format(output*100))
if __name__ == '__main__':
    app.run(debug=True)
