# pipeline_search.py
import numpy as np
import pandas as pd 

from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def quadratic_weighted_kappa(y_true , y_pred):
    '''
    Create a wrapper for cohen's kappa with quadratric weights

    '''
    return metrics.cohen_kappa_score(
        y_true,
        y_pred,
        weights='quadratic'
    )

if __name__ == "__main__":
    # load the training file
    train = pd.read_csv('dataset/train.csv')

    # we donot need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id' , axis = 1)
    test = test.drop('id' , axis = 1)

    # create labels drop useless columns

    y = train.relevance.values

    # do some lambda magic on text columns
    traindata = list(
        train.apply(lambda x : "%s %s" %(x['text1'] , x['text2']) , axis=1)
    )
    testdata = list(
        test.apply(lambda x : '%s %s'%(x['text1'] , x[text2]) , axis = 1)
    )

    # tfidf vectorizer
    tfv = TfidfVectorizer(
        min_df=3,
        max_features= None,
        strip_accents= 'unicode',
        analyzer= 'word',
        token_pattern=r'\w{1,}',
        ngram_range=(1,3),
        use_idf =1,
        sublinear_tf=1,
        stop_words='english'
    )

    # fit tfidf
    tfv.fit(traindata)
    x = tfv.transform(traindata)
    x_test = tvf.transform(testdata)

    # Initialize SVD
    svd = TruncatedSVD()

    # Initialize the standard scaler
    scl = StandardScaler()

    # we will use SVM here..
    svm_model = SVC()

    # Create the pipeline
    clf = pipeline.Pipeline(
        [
            ('svd' , svd),
            ('scl' , scl),
            ('svm' , svm_model)
        ]
    )

    # Create a parameter grid to search for best parameters for everything in the pipeline

    param_grid = {
        'svd__n_components': [200 , 300],
        'svm__C': [10, 12]
    }

    # Kappa Scorer
    kappa_scorer = metrics.make_scorer(
        quadratic_weighted_kappa,
        greater_is_better= True 
    )

    # Initialize Grid Search Model

    model = model_selection.GridSearchCV(
        estimator= clf,
        param_grid= param_grid,
        scoring=kappa_scorer,
        verbose=10,
        n_jobs = -1,
        refit=True,
        cv= 5
    )

    # fit grid search model
    model.fit(x , y)
    print('Best Score: %0.3f'%model.best_score_)
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("%s : %r" %(param_name , best_parameters[param_name]))

    # get best model
    best_model = model.best_estimator_

    # fit model with best paramteres optimized for QWK
    best_model.fit(x , y)
    pred = best_model.predict()