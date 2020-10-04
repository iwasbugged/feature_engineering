# rf_gp_minimize.py
import numpy as np
import pandas as pd 

from functools import partial

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from skopt import gp_minimize
from skopt import space

def optimize(params , param_names, x, y):
    """
    The main optimization function.
    This function takes all the arguments from the search space and training 
    features and targets. it then intializes the model by setting the chosen parameters and runs 
    cross-validation and returns a negative accuracy score
    :param Params: list of params from gp_minimize
    :param param_names : list of param names. order is important!
    :param x : training data
    :param y : labels/targets
    : return : negetive accuracy after 5 folds
    """

    # convert params to dictionary
    params = dict(zip(param_names , params))

    # initialize model with currect parameters
    model = ensemble.RandomForestClassifier(**params)

    # initialize stratified k-folds
    kf = model_selection.StratifiedKFold(n_splits=5)

    # intialize accuracy list
    accuracies = []

    # loop over all folds
    for idx in kf.split(X = x , y=y):
        train_idx , test_idx = idx[0] , idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        # fit model for current fold
        model.fit(xtrain , ytrain)

        # create predictions
        preds = model.predict(xtest)

        # calculate and append accuracy
        fold_accuracy = metrics.accuracy_score(
            ytest,
            preds
        )
        accuracies.append(fold_accuracy)

    # return negative accuraccy
    return -1 * np.mean(accuracies)


if __name__ == "__main__":
    # read the training data
    df = pd.read_csv('dataset/train.csv')

    # here we have training features
    x = df.drop('price_range' , axis = 1).values
    # and the targets
    y = df.price_range.values

    # define a parameter sapce
    param_space = [
        # max_depth is an integer between 3 and 10
        space.Integer(3,15, name='max_depth'),
        # n_estimator is an integer between 50 and 1500
        space.Integer(100, 1500 , name="n_estimators"),
        # criterion is a category. here we define list of categories
        space.Categorical(['gini' , 'entropy'] ,  name='criterion'),
        # you can also have real numbered sapce and define a 
        # distribution you want to pick it from
        space.Real(0.01 , 1 , prior='uniform' , name='max_features')
    ]

    # make a list of params names 
    # this has to be same order as the search sapce
    # inside the main function
    param_names = [
        'max_depth',
        'n_estimators',
        'criterion',
        'max_features'
    ]

    """
    BY using functools partial, i am creating a new function
    which has same parameters as the optimize function expect for the 
    fact that only one param, i.e. the "params" parameter is required
    this is how gp_minimize expects the opitimization function to be 
    you can get rid of this by reading data inside the optimize
    function or by defining the optimize function here.
    """

    optimization_function = partial(
        optimize,
        param_names = param_names,
        x =x,
        y = y
    )
    

    # NOw we call gp_minimize from scikit-optimize
    # gp_minimize uses bayesian optimization for
    # minimization of the optimization function.
    # we need a space of parameters , the function
    # itself, the number of calls/iterations we want to have

    result = gp_minimize(
        optimization_function,
        dimensions= param_space,
        n_calls= 15,
        n_random_starts=10,
        verbose=10
    )

    # create best dict and print it
    best_params = dict(
        zip(
            param_names,
            result.x
        )
    )

    print(best_params)