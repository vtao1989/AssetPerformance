## package and data
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import joblib

## all information for multiple years and multiple locations
all_info = pd.read_csv('all_info.csv')

## change location and road class to dummy variables with one hot encoding
all_info = pd.get_dummies(all_info, columns = ['location', 'roadclass'])

## select observations for all-year, one-year, two-year, three-year, and four-year models
## n=0 means all-year observations
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

## model parameters
hgbm_para = {'learning_rate':[0.1, 0.01, 0.001],
                'max_iter': [100, 200, 300, 400, 500, 600, 700],
                'max_depth': [3, 5, 7, 10]}
refit = 'neg_root_mean_squared_error'
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=2023)

## all-year PCR model
Xs = pre_data(0)
X = Xs.drop(columns=['segID', 'date1', 'date2', 'rating2', 'class1', 'class2', 'DiffRating'])
y = Xs['rating2']
hgbm_fit = GridSearchCV(HistGradientBoostingRegressor(random_state=2023), hgbm_para, refit=refit, n_jobs = -1, cv=rkf)
hgbm_fit.fit(X, y)
hgbm_best_fit = hgbm_fit.best_estimator_
joblib.dump(hgbm_best_fit, 'all_year_rating_hgbm.sav')

## all-year PCR change model
Xs = pre_data(0)
X = Xs.drop(columns=['segID', 'date1', 'date2', 'rating2', 'class1', 'class2', 'DiffRating'])
y = Xs['DiffRating']
hgbm_fit = GridSearchCV(HistGradientBoostingRegressor(random_state=2023), hgbm_para, refit=refit, n_jobs = -1, cv=rkf)
hgbm_fit.fit(X, y)
hgbm_best_fit = hgbm_fit.best_estimator_
joblib.dump(hgbm_best_fit, 'all_year_rating_difference_hgbm.sav')