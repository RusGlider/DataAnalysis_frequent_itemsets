from sklearn import tree
from task3 import *

"""
варьируя следующие параметры: 
критерий выбора атрибута разбиения (information gain, index gini) и 
соотношение мощностей обучающей и тестовой выборок (от 60%:40% до 90%:10% с шагом 5%)
"""
def experiment_tree_gini(x_train, x_test, y_train):
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    visualize_tree(clf,title='index gini')
    return y_pred

def experiment_tree_gain(x_train, x_test, y_train):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    visualize_tree(clf,title='information gain')
    return y_pred

def visualize_tree(clf,title):


    plt.figure(figsize=(20,20))
    plt.title(title)
    tree.plot_tree(clf, fontsize=7)
    plt.show()

if __name__ == '__main__':
    x,y = load_dataset_apple_quality()

    #clf = tree.DecisionTreeClassifier(criterion='gini')
    #clf = clf.fit(x, y)

    #visualize_tree(clf)

    #TODO сделать визуализацию дерева
    test_sizes = [0.80]
    classification_experiment(x, y, experiment_tree_gini, algname='tree gini index', random_state=0, test_sizes=test_sizes)
    #classification_experiment(x, y, experiment_tree_gain, algname='tree information gain', random_state=0, test_sizes=test_sizes)