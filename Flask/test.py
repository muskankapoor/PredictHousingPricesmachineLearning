from flask import Flask, render_template, url_for, request #,redirect
app = Flask(__name__)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

ames_train_data = pd.read_csv('train.csv')

# Define and remove Y from data
Y = ames_train_data['SalePrice']
ames_train_data.drop(["SalePrice"], axis=1, inplace=True)

# Test on neighborhood only
labelEncoder = preprocessing.LabelEncoder()

ames_train_data['Neighborhood'] = labelEncoder.fit_transform(ames_train_data['Neighborhood'])

x = ames_train_data[['Neighborhood']]

# Split

x_train, x_val,y_train,y_val = train_test_split(x, Y, test_size= .2, random_state=0)

# Decision Tree
decTreeReg = DecisionTreeRegressor(random_state = 0)
decTreeReg.fit(x_train, y_train)

# Random Forest
randomForestReg = RandomForestRegressor(random_state = 0)
randomForestReg.fit(x_train, y_train)

# Lasso
lasso1=Lasso(alpha=0.05,normalize=True)
lasso1.fit(x_train,y_train)

# Ridge
ridge1=Ridge(alpha=0.05,normalize=True)
ridge1.fit(x_train,y_train)

@app.route('/')
@app.route('/homepage',methods=['GET','POST'])
def home():
	return render_template('summerproject.html')
    # return render_template('home.html' )

@app.route('/predict')
def predict():
	return render_template('predict.html')

@app.route('/aboutus')
def aboutus():
	return render_template('aboutus.html')

@app.route('/answer',methods=['POST'])
def answer():
    neighborhood= float(request.form['Neighborhood'])
    predDT = decTreeReg.predict([[neighborhood]])
    predRF = randomForestReg.predict([[neighborhood]])
    predL = lasso1.predict([[neighborhood]])
    predR = ridge1.predict([[neighborhood]])
    return render_template('predictionAnswer1.html', predDT=predDT[0], predRF=predRF[0], predL=predL[0], predR=predR[0])


if __name__=="__main__":
	app.run(debug=True)
