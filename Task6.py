from sklearn.ensemble import RandomForestClassifier
from task3 import *


def random_forest_wrapper(n_estimators):
    def experiment_random_forest_n(x_train, x_test, y_train):
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=2, random_state=0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        return y_pred

    return experiment_random_forest_n

if __name__ == '__main__':
    x, y = load_dataset_apple_quality()

    for estimator_count in range(50, 110, 10):
        experiment_func = random_forest_wrapper(n_estimators=estimator_count)
        classification_experiment(x, y, experiment_func, algname=f'random forest {estimator_count} estimators', random_state=0)