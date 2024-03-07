from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from task3 import *
def experiment_bagging(x_train, x_test, y_train):
    clf = BaggingClassifier(estimator=SVC(), n_estimators = 100, random_state = 0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred

def bagging_wrapper(n_estimators):
    def experiment_bagging_n(x_train, x_test, y_train):
        clf = BaggingClassifier(estimator=SVC(), n_estimators=n_estimators, random_state=0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        return y_pred

    return experiment_bagging_n

#варьируя количество участников ансамбля (от 50 до 100 с шагом 10).
if __name__ == '__main__':
    x,y = load_dataset_apple_quality()

    #classification_experiment(x, y, experiment_bagging,algname='bagging', random_state=0)
    for estimator_count in range(50,110,10):
        print(f'estimator count: {estimator_count}')
        experiment_func = bagging_wrapper(n_estimators=estimator_count)
        classification_experiment(x, y, experiment_func, algname=f'bagging {estimator_count} estimators', random_state=0)
