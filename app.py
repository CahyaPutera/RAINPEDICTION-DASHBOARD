# Import Libraries
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier

# Import Cleaned Dataset
df = pd.read_csv('df_model6.csv')

# Split data 
X = df.drop(columns=['rain'])
y = df['rain']

# Applying scaler
scaler = RobustScaler()
X = scaler.fit_transform(X.values)

# Split train, test 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)

# Oversampling
X_train_sm, y_train_sm = SMOTE(random_state = False).fit_sample(X_train, y_train)

# Applying model
model =  joblib.load('model_RFC')

# Initiate Flask
app = Flask(__name__)

# Root
@app.route('/')
def route_root():
    return render_template('home.html')

# Homepage
@app.route('/home')
def route_home():
    return render_template('home.html')

# Dataset
@app.route('/dataset')
def route_dataset():
    return render_template('dataset.html')

# About
@app.route('/about')
def route_about():
    return render_template('about.html')

# Prediction
@app.route('/predict', methods = ['GET'])
def route_predict():
    return render_template('predict.html')

# Result
@app.route('/result')
def route_visualization():
    return render_template('result.html')


@app.route('/predict', methods=['POST'])
def route_result():
    if request.method == 'POST':

        k_index = float(request.form['k_index'])
        dewpoint_1000_hpa = float(request.form['dewpoint_1000_hpa'])
        dewpoint_850_hpa = float(request.form['dewpoint_850_hpa'])
        dewpoint_500_hpa = float(request.form['dewpoint_500_hpa'])
        cross_totals_index = float(request.form['cross_totals_index'])
        showalter_index = float(request.form['showalter_index'])
        lifted_index = float(request.form['lifted_index'])
        convective_available_potential_energy = float(request.form['convective_available_potential_energy'])
        dewpoint_700_hpa = float(request.form['dewpoint_700_hpa'])

        prob = round(model.predict_proba([[dewpoint_1000_hpa,dewpoint_850_hpa,k_index,showalter_index,dewpoint_500_hpa,lifted_index,convective_available_potential_energy,cross_totals_index,dewpoint_700_hpa]])[0][1]*100, 2)

        result = 'THE PROBABILITY OF RAINING IS {} %'.format(prob) 

        return render_template('result.html', results = result)

if __name__ == "__main__":
    app.run(debug = True)