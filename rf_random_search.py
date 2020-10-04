# rf_random_search.py
import pandas as pd 
import numpy as np 

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv('dataset/train.csv')

    # features are al columns without price_range

    # here we have training features
    x = df.drop('price_range' , axis = 1).values
    # and the target
    y = df.price_range.values

    # define the model
    classifier = ensemble.RandomForestClassifier(n_jobs = -1)

    # define a grid of parameters, this can be a dictionary or a list of dictionaries
    params = {
        'n_estimators': np.arange(100, 1500 , 100),
        'max_depth': np.arange(1, 31),
        'criterion': ['gini', 'entropy']
    }

    '''
    intialize random search 
    estimator is the model that we have defined 
    param_distributions is the grid of parameters
    we use accuracy as our metric.
    higher value of verbose implies a lot of details are printed
    cv = 5 means that we are using 5 folds cv
    n_iter is the number of iterations we want
    if param_distributions has alll the values as list,
    random search will be done by sampling without replacement
    if any of the parameters come from a distribution,
    random search uses sampling with repalcement
    '''
    model = model_selection.RandomizedSearchCV(
        estimator= classifier,
        param_distributions= params,
        n_iter= 20,
        scoring='accuracy',
        verbose=10,
        n_jobs= -1,
        cv = 5
    )

    # fit the model and extract best score
    model.fit(x ,y)
    print(f'Best score : {model.best_score_}')

    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print(f'\t {param_name} : {best_parameters[param_name]}')