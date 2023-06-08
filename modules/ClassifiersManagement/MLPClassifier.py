from sklearn.neural_network import MLPClassifier  as Classifier
import pickle
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.getcwd())
import modules.ClassifiersManagement.helper as classifierHelper
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array

class MLPClassifier:
    def __init__(self) -> None:
        pass

    def get_classifier():
        try:
            clf = pickle.load(open(classifierHelper.get_special_name(
                'pickled', 'MLPClassifier', '.pickle'), "rb"))
        except:
            feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
            X = feature_emotion_X_Y_array['X']
            y = feature_emotion_X_Y_array['y']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=para.test_size, random_state=0)

            clf = Classifier(solver='adam', max_iter=500, learning_rate='adaptive', hidden_layer_sizes=(256, 256), batch_size='auto', alpha=0.005, activation='tanh')

            clf.fit(X=X_train, y=y_train)
            pickle.dump(clf, open(classifierHelper.get_special_name(
                'pickled', 'MLPClassifier', '.pickle'), "wb"))
        return clf


    def get_classifier_through_randomized_search_cv():
        from sklearn.model_selection import RandomizedSearchCV

        try:
            random_search = pickle.load(open(classifierHelper.get_special_name(
                'pickled', 'RandomizeSearch_MLPClassifier', '.pickle'), "rb"))
        except:
            feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
            X = feature_emotion_X_Y_array['X']
            y = feature_emotion_X_Y_array['y']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=para.test_size, random_state=0)

            # Define the parameter grid to search over
            param_dist = {
                'hidden_layer_sizes': [(64,), (128,), (256,), (512,), (64, 64), (128, 128), (256, 256), (512, 256), (512, 512)],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'batch_size': ['auto'],
                'activation': ['relu', 'logistic', 'tanh'],
                'solver': ['adam', 'sgd'],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [100, 200, 500, 1000],
                'random_state': [42],
            }

            # Create an MLPClassifier object
            clf = Classifier()

            # Create a RandomizeSearchCV object
            random_search = RandomizedSearchCV(
                clf, param_distributions=param_dist, n_iter=10, n_jobs=-1, cv=5, verbose=3)

            # Fit the grid search object to the data
            random_search.fit(X=X_train, y=y_train)
            pickle.dump(random_search, open(classifierHelper.get_special_name(
                'pickled', 'RandomizeSearch_MLPClassifier', '.pickle'), "wb"))
        # Print the best parameters and best score
        print("Best parameters: ", random_search.best_params_)
        print("Best score: ", random_search.best_score_)

        return random_search


    def predict(path):
        clf = MLPClassifier.get_classifier_through_randomized_search_cv()
        # clf = MLPClassifier.get_classifier()
        return classifierHelper.predict(path, clf)


if __name__ == '__main__':
    # clf = get_classifier()
    clf = MLPClassifier.get_classifier_through_randomized_search_cv()

    feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
    X = feature_emotion_X_Y_array['X']
    y = feature_emotion_X_Y_array['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=para.test_size, random_state=0)

    y_prediction = clf.predict(X_test)
    classifierHelper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name='RandomizeSearch_MLPClassifier',title=f"Accuracy: {clf.score(X_test, y_test)} - RandomizeSearch_MLPClassifier\n{classifierHelper.get_special_name(folder_name='',prefix='')}\n{clf.best_params_}")
