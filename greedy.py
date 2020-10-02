# greedy.py
import pandas as pd 
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:
    """
    A simple and custom class for greedy feature selection, you will need to 
    to modify it quite a bit to make it suitable for your dataset
    """
    def evaluate_score(self , x ,y):
        """
        This function evaluates model on data and returns Area Under ROC Curve (AUC)
        NOTE: we fit the data and evaluate AUC on same data .
        WE ARE OVER FITTING HERE.
        But this is also a way to achieve greedy selection.
        k-fold will take k time longer.

        If you want to implement it in really  correct way, calculate OOF AUC and return mean
        AUC over k folds. This require only a few lines of change.

        :params x : training data
        :params y: targets
        :returns : overfitted area under curve the roc curve
        """
        # fit the logistic regression model,
        # and calculate AUC on the same data
        # again: BEWARE
        # you can choose any model that suits your data
        model = linear_model.LogisticRegression()
        model.fit(x,y)
        predictions = model.predict_proba(x)[:,1]
        auc = metrics.roc_auc_score(y , predictions)

        return auc

    def _feature_selection(self , x ,y):
        """
        This function does the actual greedy selction
        :params x : data,numpy array
        :params y : targets, numpy array
        :return : (best scores ,  best features)
        """
        # initialize good features list
        # and best scores to keep track of both
        good_features = []
        best_scores = []

        # calculating the number of features
        num_features = x.shape[1]

        # infinite loop
        while True:
            # intialize best feature and score of this loop
            this_feature = None
            best_score = 0

            # loop over all features
            for feature in range(num_features):
                # if feature is already in good features,
                # skip this for loop
                if feature in good_features:

                    continue
                # selected features are all good till now
                # and current feature
                selected_features = good_features + [feature]
                # remove all other feature from the data
                xtrain = x[: , selected_features]
                # calculate the score , in our case AUC
                score = self.evaluate_score(xtrain , y)
                # if score is greater then the best score
                # of this loop, change best score and best feature
                if score > best_score:
                    this_feature = feature
                    best_score = score

            # if we have  selected a feature , add it to
            # the good feature list and update best score list
            if this_feature != None:
                good_features.append(this_feature)
                best_scores.append(best_score)

            # if we did not improve during the last two rounds,
            # exit the while loop
            if len(best_score) > 2:
                if best_scores[-1] < best_scores[-2]:
                    break

        # return the best score and good features
        # why do we remove the last data point?
        return best_scores[:-1] , good_features[:-1]

    def __call__(self , x,y):
        """
        Call function will call the class on a set of arguments
        """
        # selcet features ,  return scores and selected indices
        scores , features = self._feature_selection(x , y)
        # transform data with selected features
        return x[:,features] , scores

if __name__ == "__main__":
    # generate binary classification data
    x ,y = make_classification(n_samples=1000 , n_features= 100)

    # transform data by greedy feature selection 
    x_transformed , scores = GreedyFeatureSelection()(x,y)
    print(scores)
    print(x_transformed.shape)