from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler


app = Flask(__name__, template_folder='templates')
model = pickle.load(open('heart_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


scaler = StandardScaler()
df_num = pd.read_csv('Datasets/numerical_data.csv')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
                
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        exang = int(request.form['exang'])
        trestbps = float(request.form['trestbps'])
        thalach = float(request.form['thalach'])

        if sex==0:
            sex_0 = 1
            sex_1 = 0
        elif sex==1:
            sex_0 = 0
            sex_1 = 1
                        
        if cp==0:
            cp_0 = 1
            cp_1 = 0
            cp_2 = 0
            cp_3 = 0
        elif cp==1:
            cp_0 = 0
            cp_1 = 1
            cp_2 = 0
            cp_3 = 0
        elif cp==2:
            cp_0 = 0
            cp_1 = 0
            cp_2 = 1
            cp_3 = 0
        elif cp==3:
            cp_0 = 0
            cp_1 = 0
            cp_2 = 0
            cp_3 = 1
        
        if exang==0:
            exang_0 = 1
            exang_1 = 0
        elif exang==1:
            exang_0 = 0
            exang_1 = 1
           
        df_num.loc[-1] = [age, trestbps, thalach]
        scaled_data = scaler.fit_transform(df_num)
        scaled_num = scaled_data[-1,:]
        
        output = model.predict([[scaled_num[0], scaled_num[1], scaled_num[2], sex_0, sex_1, cp_0, cp_1, cp_2, cp_3, exang_0, exang_1]])
            
        if output==1:
            return render_template('index.html',prediction_text="Warning, you are in high risk of having heart disease!")
        else:
            return render_template('index.html',prediction_text="Congratulations, you are in low risk of having heart disease:)")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

