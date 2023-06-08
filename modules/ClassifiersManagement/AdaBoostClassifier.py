from sklearn.ensemble import AdaBoostClassifier  as Classifier
import pickle
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.getcwd())
import parameters as para
import modules.ClassifiersManagement.helper as classifierHelper
sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array

class AdaBoostClassifier:
    def __init__(self) -> None:
        pass
        # print()
        
    def get_classifier():
        try:
            clf = pickle.load(open(classifierHelper.get_special_name(
                'pickled', 'AdaBoostClassifier', '.pickle'), "rb"))
        except:
            feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
            X = feature_emotion_X_Y_array['X']
            y = feature_emotion_X_Y_array['y']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=para.test_size, random_state=0)

            clf = Classifier(
                algorithm='SAMME', learning_rate=0.8, n_estimators=60)

            clf.fit(X=X_train, y=y_train)
            pickle.dump(clf, open(classifierHelper.get_special_name(
                'pickled', 'AdaBoostClassifier', '.pickle'), "wb"))
        # print('hello')
        return clf


    def get_classifier_through_randomized_search_cv():
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import RandomizedSearchCV

        try:
            random_search = pickle.load(open(classifierHelper.get_special_name(
                'pickled', 'RandomizeSearch_AdaBoostClassifier', '.pickle'), "rb"))
        except:
            feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
            X = feature_emotion_X_Y_array['X']
            y = feature_emotion_X_Y_array['y']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=para.test_size, random_state=0)

            from sklearn.tree import DecisionTreeClassifier
            # Define the parameter grid to search over
            param_dist = {
                'n_estimators': [50, 100, 200, 400],
                'learning_rate': [0.1, 0.5, 1.0],
                'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)],
                'algorithm': ['SAMME', 'SAMME.R'],
                'random_state': [42]
            }

            # Create an MLPClassifier object
            clf = Classifier()

            # Create a RandomizeSearchCV object
            random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, n_jobs=-1, cv=5, verbose=3)

            # Fit the grid search object to the data
            random_search.fit(X=X_train, y=y_train)
            pickle.dump(random_search, open(classifierHelper.get_special_name('pickled', 'RandomizeSearch_AdaBoostClassifier', '.pickle'), "wb"))
        # Print the best parameters and best score
        print("Best parameters: ", random_search.best_params_)
        print("Best score: ", random_search.best_score_)

        return random_search


    def predict(path):
        clf = AdaBoostClassifier.get_classifier_through_randomized_search_cv()
        # clf = AdaBoostClassifier.get_classifier()
        return classifierHelper.predict(path, clf)

if __name__ == '__main__':
    # clf = AdaBoostClassifier.get_classifier()
    clf = AdaBoostClassifier.get_classifier_through_randomized_search_cv()

    feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
    X = feature_emotion_X_Y_array['X']
    y = feature_emotion_X_Y_array['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=para.test_size, random_state=0)

    y_prediction = clf.predict(X_test)
    classifierHelper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name='RandomizeSearch_AdaBoostClassifier',
                            title=f"Accuracy: {clf.score(X_test, y_test)} - RandomizeSearch_AdaBoostClassifier\n{classifierHelper.get_special_name(folder_name='',prefix='')}\n{clf.best_params_}")
