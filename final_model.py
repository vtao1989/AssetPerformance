## package and data
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import time
import sendmail

## define scaling
scaler = StandardScaler()

start_time = time.time()

now = time.localtime()
print('Start at '+time.strftime("%H:%M:%S", now))

## all information for multiple years and multiple locations
all_info = pd.read_csv('all_info.csv')

## change location and road class to dummy variables with one hot encoding
all_info = pd.get_dummies(all_info, columns = ['location', 'roadclass'])


def pre_data(n):
    X=all_info
    if(n==0):
        Xs=X
    elif(n<4):
        upper = n*365
        down = (n-1)*365
        Xs = X[(X['DiffDay']<=upper) & (X['DiffDay']>down)]
    else:
        down = (n-1)*365
        Xs = X[X['DiffDay']>down]

    return(Xs)

hgbm_para = {'learning_rate':[0.1, 0.01, 0.001],
                'max_iter': [100, 200, 300, 400, 500, 600, 700],
                'max_depth': [3, 5, 7, 10]}
ann_pipe = Pipeline([("scaler", StandardScaler()),
                ("MLPRegressor", MLPRegressor(random_state=2022, max_iter=500))
                ])
ann_para = {'MLPRegressor__alpha':[0.01, 0.001, 0.0001],
            'MLPRegressor__hidden_layer_sizes': [(10,), (30,), (50,), (100,), (200,)]}
refit = 'neg_root_mean_squared_error'

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=2023)

## all years rating model
Xs = pre_data(0)
X = Xs.drop(columns=['segID', 'date1', 'date2', 'rating2', 'class1', 'class2', 'DiffRating'])
y = Xs['rating2']
hgbm_fit = GridSearchCV(HistGradientBoostingRegressor(random_state=2023), hgbm_para, refit=refit, n_jobs = -1, cv=rkf)
hgbm_fit.fit(X, y)
hgbm_best_fit = hgbm_fit.best_estimator_
joblib.dump(hgbm_best_fit, 'all_year_rating_hgbm.sav')
now = time.localtime()
print('All year rating model finished at '+time.strftime("%H:%M:%S", now))

## all years rating difference model
Xs = pre_data(0)
X = Xs.drop(columns=['segID', 'date1', 'date2', 'rating2', 'class1', 'class2', 'DiffRating'])
y = Xs['DiffRating']
hgbm_fit = GridSearchCV(HistGradientBoostingRegressor(random_state=2023), hgbm_para, refit=refit, n_jobs = -1, cv=rkf)
hgbm_fit.fit(X, y)
hgbm_best_fit = hgbm_fit.best_estimator_
joblib.dump(hgbm_best_fit, 'all_year_rating_difference_hgbm.sav')
now = time.localtime()
print('All year rating difference model finished at '+time.strftime("%H:%M:%S", now))

end_time = time.time()
elapsed_time = round((end_time-start_time)/3600, 2)
message = 'The task is finished!\n'+'Execution time:'+str(elapsed_time)+'hours'
sendmail.send_email(message)