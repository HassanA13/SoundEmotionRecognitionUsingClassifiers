from sklearn.ensemble import BaggingClassifier  as Classifier
import pickle
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.getcwd())
import parameters as para
import modules.ClassifiersManagement.helper as classifierHelper
sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array

class BaggingClassifier:
    def __init__(self) -> None:
        pass

    def get_classifier():
        try:
            clf = pickle.load(open(classifierHelper.get_special_name(
                'pickled', 'BaggingClassifier', '.pickle'), "rb"))
        except:
            feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
            X = feature_emotion_X_Y_array['X']
            y = feature_emotion_X_Y_array['y']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=para.test_size, random_state=0)

            clf = Classifier(n_estimators=60, max_samples=1., max_features=2,)

            clf.fit(X=X_train, y=y_train)
            pickle.dump(clf, open(classifierHelper.get_special_name(
                'pickled', 'BaggingClassifier', '.pickle'), "wb"))
        return clf


    def get_classifier_through_randomized_search_cv():
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import RandomizedSearchCV

        try:
            random_search = pickle.load(open(classifierHelper.get_special_name(
                'pickled', 'RandomizeSearch_BaggingClassifier', '.pickle'), "rb"))
        except:
            feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
            X = feature_emotion_X_Y_array['X']
            y = feature_emotion_X_Y_array['y']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=para.test_size, random_state=0)

            # Define the parameter grid to search over
            param_dist = {
                'n_estimators': [10, 30, 50, 60, 100, 500],
                'max_samples': [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'max_features': [0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0],
                'bootstrap': [True, False],
                'bootstrap_features': [True, False],
                'n_jobs': [-1],
                'random_state': [42]
            }

            # Create an MLPClassifier object
            clf = Classifier()

            # Create a RandomizeSearchCV object
            random_search = RandomizedSearchCV(
                clf, param_distributions=param_dist, n_iter=10, n_jobs=-1, cv=5, verbose=3)

            # Fit the grid search object to the data
            random_search.fit(X=X_train, y=y_train)
            pickle.dump(random_search, open(classifierHelper.get_special_name(
                'pickled', 'RandomizeSearch_BaggingClassifier', '.pickle'), "wb"))
        # Print the best parameters and best score
        print("Best parameters: ", random_search.best_params_)
        print("Best score: ", random_search.best_score_)

        return random_search

    def predict(path):
        # clf = BaggingClassifier.get_classifier()
        clf = BaggingClassifier.get_classifier_through_randomized_search_cv()
        return classifierHelper.predict(path, clf)

if __name__ == '__main__':
    # clf = get_classifier()
    clf = BaggingClassifier.get_classifier_through_randomized_search_cv()

    feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
    X = feature_emotion_X_Y_array['X']
    y = feature_emotion_X_Y_array['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=para.test_size, random_state=0)

    y_prediction = clf.predict(X_test)
    classifierHelper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name='RandomizeSearch_BaggingClassifier',title=f"Accuracy: {clf.score(X_test, y_test)} - RandomizeSearch_BaggingClassifier\n{classifierHelper.get_special_name(folder_name='',prefix='')}\n{clf.best_params_}")
