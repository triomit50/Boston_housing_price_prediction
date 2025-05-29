# Libraries import

import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# App creation
app = Flask(__name__)

# Model Loading --> Model Tool
reg_model = pickle.load(open('regModel.pkl',"rb"))

# Scaler loading --> Scaler Tool
scaler = pickle.load(open("scaling.pkl","rb"))

# App routing...

# initial home page
@app.route('/')
def home():
    return render_template('home.html')

# creating predict api : sending request to the app and get the response
@app.route('/predict_api',methods = ['POST'])
def predict_api():
    data = request.json['data']     # input as 'json' format, stored in the 'data' variable
    print(data)

    # converting the entire input data into "a single row array"
    print(np.array(list(data.values())).reshape(1,-1))      

    # scaling the data --> using the scaler tool
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    
    # getting the output of the input data , using the Model Tool
    output = reg_model.predict(new_data)

    print(output[0])
    
    return jsonify(output[0])

@app.route('/predict',methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = reg_model.predict(final_input)[0]
    return render_template("home.html",prediction_text = "This is the predicted House Price : {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)

