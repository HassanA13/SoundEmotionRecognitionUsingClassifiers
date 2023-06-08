from sklearn.svm import SVC  as Classifier
import pickle
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.getcwd())
import modules.ClassifiersManagement.helper as classifierHelper
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array

class SVCClassifier:
    def __init__(self) -> None:
        pass

    def get_classifier():
        try:
            clf = pickle.load(open(classifierHelper.get_special_name(
                'pickled', 'SVCClassifier', '.pickle'), "rb"))
        except:
            feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
            X = feature_emotion_X_Y_array['X']
            y = feature_emotion_X_Y_array['y']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=para.test_size, random_state=0)

            clf = Classifier(C=3, gamma=0.01, kernel='rbf',random_state=0, max_iter=100000)

            clf.fit(X=X_train, y=y_train)

            pickle.dump(clf, open(classifierHelper.get_special_name(
                'pickled', 'SVCClassifier', '.pickle'), "wb"))
        return clf


    def get_classifier_through_randomized_search_cv():
        from sklearn.model_selection import RandomizedSearchCV

        try:
            random_search = pickle.load(open(classifierHelper.get_special_name(
                'pickled', 'RandomizeSearch_SVCClassifier', '.pickle'), "rb"))
        except:
            feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
            X = feature_emotion_X_Y_array['X']
            y = feature_emotion_X_Y_array['y']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=para.test_size, random_state=0)

            # Define the parameter grid to search over
            param_dist = {
                'C': [0.0005, 0.001, 0.002, 0.01, 0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4, 5],
                'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10],
                'coef0': [0, 0.1, 1, 10],
                'shrinking': [True, False],
                'tol': [1e-4, 1e-3, 1e-2],
                'max_iter': [100, 500, 1000],
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
                'pickled', 'RandomizeSearch_SVCClassifier', '.pickle'), "wb"))
        # Print the best parameters and best score
        print("Best parameters: ", random_search.best_params_)
        print("Best score: ", random_search.best_score_)

        return random_search


    def predict(path):
        clf = SVCClassifier.get_classifier_through_randomized_search_cv()
        # clf = SVCClassifier.get_classifier()
        return classifierHelper.predict(path, clf)


if __name__ == '__main__':
    # clf = SVCClassifier.get_classifier()
    clf = SVCClassifier.get_classifier_through_randomized_search_cv()

    feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
    X = feature_emotion_X_Y_array['X']
    y = feature_emotion_X_Y_array['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=para.test_size, random_state=0)

    y_prediction = clf.predict(X_test)
    classifierHelper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name='RandomizeSearch_SVCClassifier',
                            title=f"Accuracy: {clf.score(X_test, y_test)} - RandomizeSearch_SVCClassifier\n{classifierHelper.get_special_name(folder_name='',prefix='')}\n{clf.best_params_}")
