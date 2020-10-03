# rf_grid_search.py
import pandas as pd 
import numpy as np 
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv('dataset/train.csv')

    # features are all columns without price_range
    # Note that there is no Id column in this dataset
    # here we have training features
    x = df.drop('price_range' , axis=1).values 
    # and the targets
    y = df.price_range.values 
    # define the model here
    classifier = ensemble.RandomForestClassifier(n_jobs = -1)

    # define a grid of parameters this can be a dictionary or a list of dictionary
    # 
    params_grid = {
        'n_estimators' : [100, 200, 250, 300, 400, 500],
        'max_depth': [1,2,5,7,11,15],
        'criterion': ['gini' , 'entropy'],
    }  

    # initialize grid search estimator is the model that we have defined
    # param_grid is the grid of parameters we use accuracy as our metric 
    # higher value of verbose implies a lot of details are printed
    # cv= 5 means we are using 5 folds cv

    model = model_selection.GridSearchCV(
        estimator= classifier,
        param_grid= params_grid,
        scoring='accuracy',
        verbose=10,
        n_jobs=1,
        cv=5
    )

    # fit the model and extract best score
    model.fit(x,y)
    print(f"Best score : {model.best_score_}")

    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(params_grid.keys()):
        print(f"\t{param_name} : {best_parameters[param_name]}")