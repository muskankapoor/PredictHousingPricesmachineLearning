

from flask import Flask, render_template, url_for, request #,redirect
app = Flask(__name__)

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
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

ames_train_data.drop(['Id', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'LotShape', 'LandContour', 'LandSlope', 'YrSold', 'MoSold'], axis=1, inplace=True)

ames_train_data.drop(['MiscFeature', 'Alley', 'LotFrontage'], axis=1, inplace=True)

ames_train_data_numerical = ames_train_data.select_dtypes([np.number]).columns
# for i in ames_train_data_numerical:
#     median=ames_train_data[i].mode()
#     ames_train_data[i].fillna(median)

for i in ames_train_data.columns:
    ames_train_data.dropna(subset=[i], inplace=True)  #drop NA (missing data) in the column,
    
# Encode non-numerical and numerical categorical data:

ames_train_data_categorical = ames_train_data.select_dtypes(include=['object']).columns
ames_train_data = pd.get_dummies(ames_train_data)

# normalize LotArea,TotalBsmtSF,1stFlrSF,GrLivArea,TotRmsAbvGrd
log=["LotArea","TotalBsmtSF","1stFlrSF","GrLivArea","TotRmsAbvGrd"]
for i in log:
    ames_train_data[i] = np.log1p(ames_train_data[i])
    
# Define and remove Y from data
Y = ames_train_data['SalePrice']
ames_train_data.drop(["SalePrice"], axis=1, inplace=True)

# Test on neighborhood only

x = ames_train_data['Neighborhood']

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
	return render_template('predictionAnswer.html')

@app.route('/aboutus')
def aboutus():
	return render_template('aboutus.html')


def run_Predictions(neighborhood):
    predDT = decTreeReg.predict(x_val)
    predRF = randomForestReg.predict(x_val)
    predL = lasso1.predict(x_val)
    predR = ridge1.predict(x_val)

@app.route('/answer',methods=['POST'])
def answer():


	neighborhood=request.form['Neighborhood'];

    
	return  render_template('predictionAnswer1.html', predDT, predRF, predL, predR)


if __name__=="__main__":
	app.run(debug=True)





