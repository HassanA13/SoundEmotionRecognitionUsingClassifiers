import pickle
import sys
import os

class CatBoostFeatureSelector:
    def get_selector():
        sys.path.insert(0, os.getcwd()+"\modules\ClassifiersManagement")
        import modules.ClassifiersManagement.helper as classifierHelper
        import CatBoost as cb
        # import pandas as pd
        try:
            clf = pickle.load(open(classifierHelper.get_special_name(
                'pickled', 'CatBoostFeatureSelector', '.pickle'), "rb"))
        except:
            sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
            from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array
            feature_emotion_X_Y_array = extract_feature_emotion_X_y_array(filter=False)
            X = feature_emotion_X_Y_array['X']
            y = feature_emotion_X_Y_array['y']

            # Create a CatBoostClassifier model
            clf = cb.CatBoostClassifier(
                                        depth=5,
                                        )
            # Fit the model to the data
            clf.fit(X, y)
            pickle.dump(clf, open(classifierHelper.get_special_name(
                'pickled', 'CatBoostFeatureSelector', '.pickle'), "wb"))
        return clf
    
    def get_feature_importance():
        model = CatBoostFeatureSelector.get_selector()
        feature_importance = model.get_feature_importance()
        return feature_importance

    def get_boolean_array_for_most_important_feature(ratio=0.1):
        feature_importance = CatBoostFeatureSelector.get_feature_importance()
        return feature_importance >= max(feature_importance)*ratio
    
    def filter_features(X):
        import numpy as np
        # Convert the input list to a numpy array
        X = np.array(X)
        y = CatBoostFeatureSelector.get_boolean_array_for_most_important_feature()
        return X[:,y]

if __name__ == '__main__':
    sys.path.insert(0, os.getcwd()+"\modules\ClassifiersManagement")
    import modules.ClassifiersManagement.helper as classifierHelper
    import catboost as cb
    sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
    from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array

    feature_emotion_X_Y_array = extract_feature_emotion_X_y_array(filter=False)
    X = feature_emotion_X_Y_array['X']
    y = feature_emotion_X_Y_array['y']
    # Create a CatBoostClassifier model
    clf = cb.CatBoostClassifier()

    # Fit the model to the data
    clf.fit(X, y)
    pickle.dump(clf, open(classifierHelper.get_special_name('pickled', 'CatBoostFeatureSelector', '.pickle'), "wb"))