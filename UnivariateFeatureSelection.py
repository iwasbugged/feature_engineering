# UnivariateFeatureSelection
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

class UnivariateFeatureSelection:
    def __init__(self , n_features , problem_type , scoring):
        """
        Custom univariate feature selection wrapper on different univariate
        feature selection models from scikit-learn.
        :params n_features: SelectPercetile if float else SelectKBest
        :params problem_type: classifiction or regression
        :params scoring: scoring function, string
        """
        # for a given problem type , there are only a few valid scoring
        # methods you can extend this with your own custom methods if you wish
        if problem_type == 'classification':
            valid_scoring = {
                'f_classif' : f_classif,
                'chi2' : chi2,
                'mutual_info_classif' : mutual_info_classif
            }
        else:
            valid_scoring = {
                'f_regression': f_regression,
                'mutual_info_regression': mutual_info_regression
            }

        # raise exeption if we do not have a valid scoring method
        if scoring not in valid_scoring:
            raise Exception('Invalid scoringfunction')

        # if n_features is int , we use selectkbest
        # if n_features is float , we use selectpercentile
        # please note that it is int in both cases in sklearn
        if isinstance(n_features , int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k= n_features
            )
        elif isinstance(n_features , float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features * 100)
            )
        else:
            raise Exception('Invalid type of feature')

    # same fit function
    def fit(self , x ,y):
        return self.selection.fit(x,y)

    # same transform funtion
    def transform(self , x):
        return self.selection.transform(x)

    # same fit_transform function
    def fit_transform(self , x ,y):
        return self.selection.fit_transform(x , y)

