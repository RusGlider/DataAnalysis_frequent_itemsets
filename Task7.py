from sklearn.ensemble import GradientBoostingClassifier
from task3 import *


def boosting_wrapper(n_estimators):
    def experiment_boosting_n(x_train, x_test, y_train):
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1.0, max_depth=1, random_state=0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        return y_pred

    return experiment_boosting_n

if __name__ == '__main__':
    x, y = load_dataset_apple_quality()

    for estimator_count in range(50, 110, 10):
        experiment_func = boosting_wrapper(n_estimators=estimator_count)
        classification_experiment(x, y, experiment_func, algname=f'boosting {estimator_count} estimators', random_state=0)