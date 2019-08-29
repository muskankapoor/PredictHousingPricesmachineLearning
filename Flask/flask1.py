

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

# code from cocalc:
train = pd.read_csv('train.csv')
train.drop(['Id', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'LotShape', 'LandContour', 'LandSlope', 'YrSold', 'MoSold'], axis=1, inplace=True)
printA = train.columns

# Check Missing Data:
missing_Data=train.isnull().sum().sort_values(ascending = False).head(10)

# Data Preprocessing - Step 2: Remove Columns With High Percentage of Missing Data
train.drop(['MiscFeature', 'Alley', 'LotFrontage'], axis=1, inplace=True)

# ---------------------------------------------------------------------------------------------------------------
# Data Preprocessing - Step 3: Remove Rows with Missing Values
ames_train_data_numerical = train.select_dtypes([np.number]).columns
for i in ames_train_data_numerical:
    median=train[i].mode()
    train[i].fillna(median)
    
for i in train.columns:
    train.dropna(subset=[i], inplace=True)  #drop NA (missing data) in the column,



final_T=train

 # ---------------------------------------------------------------------------------


def takeCoef(elem):
    return elem[1]

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


	neighborhood=request.form['Neighborhood'];
	# bed=request.form['bed']
	new_T=final_T[final_T['Neighborhood'] == neighborhood ]
	# if bed=="Any_is_Good":
	# 	new_T=final_T[final_T['Neighborhood'] == neighborhood ]
	# else:
	# 	new_T=final_T[final_T['Neighborhood'] == neighborhood ]
	# 	new_T=new_T[new_T['BedroomAbvGr'] == int(bed)]

	

	samples = new_T.shape[0]

	

	#  # Data Preprocessing - Step 3: Remove Rows with Missing Values
	ames_train_data_numericalA = new_T.select_dtypes([np.number]).columns
	for i in ames_train_data_numericalA:
	    median=new_T[i].mode()
	    new_T[i].fillna(median)
	    
	for i in new_T.columns:
	    new_T.dropna(subset=[i], inplace=True)  #drop NA (missing data) in the column,

	for i in new_T.columns:
		if (np.count_nonzero(new_T[i])/new_T.shape[0]) < 0.5:
			del new_T[i]
		elif new_T[i].nunique()==1:   #find wether all value in a column is same
			del new_T[i]
	# Feature Selection - Calculate and Display R^2 For Each


	correlations2 = []
	ames_train_data_numericalA = new_T.select_dtypes([np.number]).columns

	for i in ames_train_data_numericalA:
	    coef = np.corrcoef(new_T[i], new_T['SalePrice'])
	    correlations2.append((i, (coef[0][1] ** 2)))

	correlations_sorted2 = sorted(correlations2, key=takeCoef, reverse=True)



	ames_train_data_categorical = new_T.select_dtypes(include=['object']).columns
	# print(len(ames_train_data_categorical))
	new_T = pd.get_dummies(new_T)

	# normalize LotArea,TotalBsmtSF,1stFlrSF,GrLivArea,TotRmsAbvGrd
	log=["LotArea","TotalBsmtSF","1stFlrSF","GrLivArea","TotRmsAbvGrd"]
	for i in log:
	    new_T[i] = np.log1p(new_T[i])

	#------------------------------------------------------------------------------------------------------------
	# Define and remove Y from data
	Y = new_T['SalePrice']
	new_T.drop(["SalePrice"], axis=1, inplace=True)

	# Define features
	# features = ['GrLivArea', 'GarageCars']

	# test all features

	x = new_T


	#split train.csv into 20 % for validation and 80% for train
	x_train, x_val,y_train,y_val = train_test_split(x, Y, test_size= .2, random_state=0)

	# print("Train set has {}".format(x_train.shape[0]))
	x_train_An=x_train.shape[0]
	# print("Validation set has {}".format(x_val.shape[0]))
	x_val_An=x_val.shape[0]

	#-------------------------------------------------------------------------------------------------------------
	# Define and Fit Models:
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

	#----------------------------------------------------------------------------------------------------
	# Make Predictions
	predDT = decTreeReg.predict(x_val)
	predRF = randomForestReg.predict(x_val)
	predL = lasso1.predict(x_val)
	predR = ridge1.predict(x_val)

	# Calculate MAE
	decisionTreeMAE = mean_absolute_error(predDT, y_val) # 23192.20895522388 # Encoding: 23046.988805970148
	randomForestMAE = mean_absolute_error(predRF, y_val) # 18686.317537313433 # Encoding: 18871.216791044775
	lassoMAE = mean_absolute_error(predL, y_val) # 19516.631671630043 # Encoding: 19758.633342932753
	ridgeMAE = mean_absolute_error(predR, y_val) # 18822.669089402567 # Encoding: 18818.902286782017

	#Calculate MAPE
	decisionTreeMAPE = np.mean(np.abs((y_val - predDT) / y_val)) * 100
	randomForestMAPE = np.mean(np.abs((y_val - predRF) / y_val)) * 100
	lassoMAPE = np.mean(np.abs((y_val - predL) / y_val)) * 100
	ridgeMAPE = np.mean(np.abs((y_val - predR) / y_val)) * 100

	return  render_template('predictionAnswer1.html',Neighboo=neighborhood, train=new_T, dataNew=new_T.to_html(), samples=samples, r_squ=correlations_sorted2, 
						x_train_An=x_train_An, x_val_An=x_val_An,  decisionTreeMAE= decisionTreeMAE, decisionTreeMAPE=decisionTreeMAPE,
    					 randomForestMAE=randomForestMAE, randomForestMAPE =randomForestMAPE, lassoMAE=lassoMAE,lassoMAPE=lassoMAPE, ridgeMAE=ridgeMAE, ridgeMAPE=ridgeMAPE)




if __name__=="__main__":
	app.run(debug=True)





