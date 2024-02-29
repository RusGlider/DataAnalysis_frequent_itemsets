#import scikit
import pandas as pd

from util import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

def calculate_metrics(y_test, y_pred):
    TP = sum(y_test[i] == 1 and y_pred[i] == 1 for i in range(len(y_test)))
    FP = sum(y_test[i] == 0 and y_pred[i] == 1 for i in range(len(y_test)))
    TN = sum(y_test[i] == 0 and y_pred[i] == 0 for i in range(len(y_test)))
    FN = sum(y_test[i] == 1 and y_pred[i] == 0 for i in range(len(y_test)))
    accuracy = (TP+TN) / (TP+FN+FP+TN) #accuracy_score(y_test, y_pred)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    F_measure = (2*precision*recall) / (precision+recall)

    return accuracy, precision, recall, F_measure

def get_accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

def get_precision():
    pass

def get_recall():
    pass

def get_Fmeasure():
    pass


def experiment_gaussian(x_train, x_test, y_train):
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    return y_pred

def experiment_bernoulli(x_train, x_test, y_train):
    bernoulli_nb = BernoulliNB()
    y_pred = bernoulli_nb.fit(x_train, y_train).predict(x_test)
    return y_pred

def experiment_f(x_train, x_test, y_train):
    bernoulli_nb = BernoulliNB()
    y_pred = bernoulli_nb.fit(x_train, y_train).predict(x_test)
    return y_pred

def classification_experiment(x, y, algorithm, algname='', random_state=42):
    test_sizes = [0.05 * i for i in range(1, 20)]  # [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    results = pd.DataFrame(columns=['test_size', 'accuracy', 'precision', 'recall', 'F_measure'])

    for test_size in test_sizes:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state) #
        # t1 = time.time()
        y_pred = algorithm(x_train, x_test, y_train)
        # t2 = time.time()

        accuracy, precision, recall, F_measure = calculate_metrics(y_test, y_pred)
        print(f'test_split: {int((1 - test_size) * 100)}:{int(test_size * 100)}')
        print(f'accuracy: {accuracy}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'F_measure: {F_measure}')
        # print(f'time taken: {t2 - t1} s')
        print()

        results.loc[len(results)] = [test_size, accuracy, precision, recall, F_measure]

    print(results)
    visualize_results(
        items=[results['accuracy'], results['precision'], results['recall'], results['F_measure']],
        supports=results['test_size'],
        title=f'Метрики алгоритма {algname} в зависимости от размера тестовой выборки',
        legend=['accuracy', 'precision', 'recall', 'F_measure'],
        xlabel='Процент тестовой выборки',
        ylabel='Точность',
        grid=True

    )

def load_dataset_apple_quality():
    df = pd.read_csv('apple_quality.csv')[:-1]  # skip last row
    # Dropping null values
    df.dropna(inplace=True)
    # dropping useless values
    df.drop(["A_id"], axis=1, inplace=True)
    df['Acidity'] = pd.to_numeric(df['Acidity'], errors='coerce')
    df.Quality = [1 if quality == "good" else 0 for quality in df.Quality]

    x_data = df.drop(["Quality"], axis=1)
    y = df.Quality.values

    # Normalization
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
    return x, y


def task3():


    df = pd.read_csv('apple_quality.csv')[:-1] #skip
    # Dropping null values
    df.dropna(inplace=True)
    # dropping useless values
    df.drop(["A_id"], axis=1, inplace=True)
    df['Acidity'] = pd.to_numeric(df['Acidity'], errors='coerce')
    df.Quality = [1 if quality == "good" else 0 for quality in df.Quality]

    x_data = df.drop(["Quality"], axis=1)
    y = df.Quality.values

    # Normalization
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

    # соотношение мощностей обучающей и тестовой выборок от 60%:40% до 90%:10% с шагом 5%.
    test_sizes = [0.05*i for i in range(1,20)]#[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    results = pd.DataFrame(columns=['test_size','accuracy', 'precision', 'recall', 'F_measure'])

    for test_size in test_sizes:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        #t1 = time.time()
        y_pred = experiment_gaussian(x_train, x_test, y_train)
        #t2 = time.time()


        accuracy, precision, recall, F_measure = calculate_metrics(y_test, y_pred)
        print(f'test_split: {int((1-test_size)*100)}:{int(test_size*100)}')
        print(f'accuracy: {accuracy}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'F_measure: {F_measure}')
        #print(f'time taken: {t2 - t1} s')
        print()

        results.loc[len(results)] = [test_size, accuracy, precision, recall, F_measure]

    print(results)
    visualize_results(
        items=[results['accuracy'], results['precision'], results['recall'], results['F_measure']],
        supports=results['test_size'],
        title='Точность в зависимости от размера тестовой выборки',
        legend=['accuracy', 'precision', 'recall', 'F_measure'],
        xlabel='Процент тестовой выборки',
        ylabel='Точность',
        grid=True

    )





if __name__ == '__main__':
    #task3()
    #print(accuracy_score([0,1,1,0],[1,1,1,0]))
    x,y = load_dataset_apple_quality()
    classification_experiment(x, y, experiment_gaussian, algname='gaussian', random_state=0)
    #classification_experiment(x, y, experiment_bernoulli, algname='bernoulli')