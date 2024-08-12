from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load models
with open('rfc_model.pkl', 'rb') as file:
    rfc_model = pickle.load(file)

with open('mlp_model.pkl', 'rb') as file:
    mlp_model = pickle.load(file)

categorical_mapping = {
    'Ya': 1,
    'Tidak': 0
}

status_mapping = {
    1: 'Tidak Terindikasi Diabetes',
    2: 'Risiko Rendah Diabetes',
    3: 'Risiko Tinggi Diabetes',
    4: 'Diabetes Melitus'
}

def predict_diabetes(polidpsia, poliuria, luka_lamban_sembuh, berat_badan_turun, gdp, gds):
    data = np.array([[
        categorical_mapping[polidpsia],
        categorical_mapping[poliuria],
        categorical_mapping[luka_lamban_sembuh],
        categorical_mapping[berat_badan_turun],
        float(gdp),
        float(gds)
    ]])

    rfc_prediction = rfc_model.predict(data)[0]
    mlp_prediction = mlp_model.predict(data)[0]

    rfc_status = status_mapping[rfc_prediction]
    mlp_status = status_mapping[mlp_prediction]

    if rfc_prediction == mlp_prediction:
        status_prediction = status_mapping[rfc_prediction]
    else:
        status_prediction = status_mapping[max(rfc_prediction, mlp_prediction)]

    return rfc_status, mlp_status, status_prediction

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET','POST'])
def prediksi():
    if request.method == 'POST':
        try:
            polidpsia = request.form.get('polidpsia')
            poliuria = request.form.get('poliuria')
            luka_lamban_sembuh = request.form.get('luka_lamban_sembuh')
            berat_badan_turun = request.form.get('berat_badan_turun')
            gdp = request.form.get('gdp')
            gds = request.form.get('gds')

            rfc_status, mlp_status, status_prediction = predict_diabetes(polidpsia, poliuria, luka_lamban_sembuh, berat_badan_turun, gdp, gds)

            return render_template('prediction.html', 
                                   rfc_status=rfc_status, 
                                   mlp_status=mlp_status, 
                                   status_prediction=status_prediction, 
                                   prediction_done=True)
        except Exception as e:
            flash(str(e))
            return redirect(url_for('home'))

    return render_template('prediction.html', prediction_done=False)

@app.route("/data")
def data():
    data = pd.read_excel('Data Januari - April 2024.xlsx')

    data_dict = data.to_dict(orient='records')
    return render_template('table.html', data_dict=data_dict)

if __name__ == '__main__':
    app.run(debug=True)
