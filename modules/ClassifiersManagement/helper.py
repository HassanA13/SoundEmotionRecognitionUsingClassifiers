import time
import os
import sys
sys.path.insert(0, os.getcwd())
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
from modules.FeaturesManagement.Feature import Feature


def get_special_name(folder_name='memory', prefix='', extension=''):
    name = prefix
    for dataset in para.datasets:
        name = name + \
            f"_{dataset}"
    for feature in para.features:
        name = name + \
            f"_{feature}"
    for emotion in para.emotions:
        name = name + \
            f"_{emotion}"

    name = os.path.join(folder_name, f"{name}{extension}")
    return name


def confusion_matrix(y_test, y_prediction, classifier_name, title='', normalize='true'):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    class_names = para.emotions

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_prediction,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )

    disp.ax_.set_title(title)

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(get_special_name(folder_name=f'statistics\\{classifier_name}', prefix=str(
        time.time())+f'_{classifier_name}', extension='.png'), bbox_inches='tight')


def predict(path, classifier):
    feature = Feature(path).extract_feature().reshape(1, -1)

    sys.path.insert(0, os.getcwd()+"\modules\FeatureSelectionManagement")
    from  modules.FeatureSelectionManagement.CatBoostFeatureSelector import CatBoostFeatureSelector 
    feature = CatBoostFeatureSelector.filter_features(feature)

    return para.emotions[int(classifier.predict(feature)[0])]
