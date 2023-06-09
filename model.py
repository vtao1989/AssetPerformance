## package and data
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

def train_model(X, y, n):

    # training parameters
    rf_para = {'n_estimators':[100, 200, 300, 400, 500, 600, 700],
            'max_depth': [3, 5, 7, 10],
            'max_features':['sqrt', 'log2', None]}
    hgbm_para = {'learning_rate':[0.1, 0.01, 0.001],
                'max_iter': [100, 200, 300, 400, 500, 600, 700],
                'max_depth': [3, 5, 7, 10]}
    ann_pipe = Pipeline([("scaler", StandardScaler()),
                    ("MLPRegressor", MLPRegressor(random_state=2022, max_iter=500))
                    ])
    ann_para = {'MLPRegressor__alpha':[0.01, 0.001, 0.0001],
                'MLPRegressor__hidden_layer_sizes': [(10,), (30,), (50,), (100,), (200,)]}
    scoring = {'r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'}
    refit = 'neg_root_mean_squared_error'

    ## model setting
    cv_r = []

    ## model training
    np.random.seed(2023)
    for i in range(n):
        ## configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True)
        cv_outer = KFold(n_splits=5, shuffle=True)
        # lm
        scores = cross_validate(LinearRegression(), X, y, scoring=scoring, cv=cv_outer, n_jobs=-1)
        cv_r.append(scores['test_r2'].mean())
        cv_r.append(scores['test_neg_mean_absolute_error'].mean())
        cv_r.append(scores['test_neg_root_mean_squared_error'].mean())    
        ## rf
        rf_fit = GridSearchCV(RandomForestRegressor(random_state=2022), rf_para, refit=refit, n_jobs = -1, cv=cv_inner)
        scores = cross_validate(rf_fit, X, y, scoring=scoring, cv=cv_outer, n_jobs=-1)
        cv_r.append(scores['test_r2'].mean())
        cv_r.append(scores['test_neg_mean_absolute_error'].mean())
        cv_r.append(scores['test_neg_root_mean_squared_error'].mean())
        ## gbm
        hgbm_fit = GridSearchCV(HistGradientBoostingRegressor(random_state=2022), hgbm_para, refit=refit, n_jobs = -1, cv=cv_inner)
        scores = cross_validate(hgbm_fit, X, y, scoring=scoring, cv=cv_outer, n_jobs=-1)
        cv_r.append(scores['test_r2'].mean())
        cv_r.append(scores['test_neg_mean_absolute_error'].mean())
        cv_r.append(scores['test_neg_root_mean_squared_error'].mean())
        ## ann
        ann_fit = GridSearchCV(ann_pipe, ann_para, refit=refit, n_jobs = -1, cv=cv_inner)
        scores = cross_validate(ann_fit, X, y, scoring=scoring, cv=cv_outer, n_jobs=-1)
        cv_r.append(scores['test_r2'].mean())
        cv_r.append(scores['test_neg_mean_absolute_error'].mean())
        cv_r.append(scores['test_neg_root_mean_squared_error'].mean())
        now = time.localtime()
        print('Round:'+str(i)+' finished at '+time.strftime("%H:%M:%S", now))

    return(cv_r)

# for i in [0,1,2,3,4]:

#     print('Session: '+str(i))

#     n=10

#     # prepare data
#     Xs = pre_data(i)
#     X = Xs.drop(columns=['segID', 'date1', 'date2', 'rating2', 'class1', 'class2', 'DiffRating'])
#     y = Xs['rating2']

#     # model training
#     cv_r = train_model(X, y, n)

#     ## save results
#     cv_result = pd.DataFrame({'value': cv_r,
#                             'model': (['lm']*3 + ['rf']*3 + ['hgbm']*3 + ['ann']*3)*n,
#                             'eval': ['r2', 'mae', 'rmse']*n*4,
#                             'year':i})
#     if(i==0):
#         cv_result.to_csv('rating_cv_result.csv', index=False)
#     else:
#         cv_result.to_csv('rating_cv_result.csv', mode='a', header=False, index=False)

for i in [0,1,2,3,4]:

    print('Session: '+str(i))

    n=10

    # prepare data
    Xs = pre_data(i)
    X = Xs.drop(columns=['segID', 'date1', 'date2', 'rating2', 'class1', 'class2', 'DiffRating'])
    y = Xs['DiffRating']

    # model training
    cv_r = train_model(X, y, n)

    ## save results
    cv_result = pd.DataFrame({'value': cv_r,
                            'model': (['lm']*3 + ['rf']*3 + ['hgbm']*3 + ['ann']*3)*n,
                            'eval': ['r2', 'mae', 'rmse']*n*4,
                            'year':i})
    if(i==0):
        cv_result.to_csv('rating_diff_cv_result.csv', index=False)
    else:
        cv_result.to_csv('rating_diff_cv_result.csv', mode='a', header=False, index=False)

end_time = time.time()
elapsed_time = round((end_time-start_time)/3600, 2)
print('Execution time:', elapsed_time, 'hours')

import sendmail
message = 'The task is finished!\n'+'Execution time:'+str(elapsed_time)+'hours'
sendmail.send_email(message)