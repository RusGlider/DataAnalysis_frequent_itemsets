from Task8_Clusterization1 import *

from sklearn.metrics import silhouette_score

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import cross_val_score
from numpy import diff

def try_cross_val_score(X, Y):
    pass

def try_silhouette_score(X, Y):
    n_clusters = np.arange(2,11)#[2, 3, 4, 5, 6, 7, 8, 9, 10]
    accuracies = []
    for n_cluster in n_clusters:
        kmeans = KMeans(n_clusters=n_cluster)
        accuracies.append(silhouette_score(X, kmeans.fit_predict(X)))

    plt.plot(n_clusters, accuracies)
    plt.title('Метод силуэта')
    plt.xlabel('n_clusters')
    plt.ylabel('score')
    plt.grid()
    plt.show()

def try_elbow_score(X, Y):
    n_clusters = np.arange(2,15)
    accuracies = []
    for n_cluster in n_clusters:
        kmeans = KMeans(n_clusters=n_cluster).fit(X)
        accuracies.append(kmeans.score(X,Y))

    plt.plot(n_clusters, accuracies)
    plt.xlabel('n_clusters')
    plt.ylabel('score')
    plt.grid()
    plt.title('Метод локтя')
    plt.show()

    df = diff(accuracies)
    df = np.append(df, df[-1])
    plt.plot(n_clusters, df)
    plt.xlabel('n_clusters')
    plt.ylabel('score')
    plt.grid()
    plt.title('Метод локтя (производная)')
    plt.show()

    df2 = diff(df)
    df2 = np.append(df2, df2[-1])
    plt.plot(n_clusters, df2)
    plt.xlabel('n_clusters')
    plt.ylabel('score')
    plt.grid()
    plt.title('Метод локтя (вторая производная)')
    plt.show()

if __name__ == '__main__':
    """
    Для набора данных из задания о разделительной кластеризации 
    подберите оптимальное количество кластеров 
    с помощью двух любых приемов из следующего множества: 
        метод локтя, 
        кросс-валидация, 
        силуэтный коэффициент, 
        визуализация матрицы схожести. 
    Постройте диаграммы, подтверждающие полученные результаты.
    """

    #n_clusters = 7
    hidden_clusters = 7
    X, y = generate_convex2D_dataset(n_blobs=hidden_clusters, n_samples_per_blob=100, random_state=2, std=[0.5,1.5])

    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=map_to_plt_colors(y))
    plt.title(f'Выпуклый датасет с {hidden_clusters} кластерами')
    plt.show()

    try_elbow_score(X, y)
    try_silhouette_score(X, y)



