from util import *
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

import matplotlib.colors as colors
from sklearn.datasets import make_blobs, make_moons, make_circles

#dataset https://www.kaggle.com/datasets/bhavikjikadara/brand-laptops-dataset
def load_dataset_laptops():
    df = pd.read_csv('laptops.csv')
    df.drop(["index"], axis=1, inplace=True)

    X = df[['Price','Rating']].values
    Y = df[['brand']].values
    Y = [item[0] for item in Y]
    return X, Y


def generate_convex2D_dataset(n_blobs=3, n_samples_per_blob=100, n_features=2,std=None, random_state=1):
    # Define the centers of the clusters
    #centers = [(-5, -5), (5, 5), (0, 0)]
    # Define the standard deviation of each cluster
    #cluster_std = [0.7, 1, 1.5]
    n_samples = n_blobs * n_samples_per_blob
    if std is None:
        std = [0.5, 1.5]
    cluster_std = np.random.uniform(low=std[0],high=std[1],size=(n_blobs,))
    # Generate the dataset
    X, Y = make_blobs(n_samples=n_samples, cluster_std=cluster_std, centers=n_blobs, n_features=n_features, random_state=random_state)

    return X, Y

def generate_moons2D_dataset(n_samples=100, noise = 0):
    X, Y = make_moons(n_samples=n_samples, noise=noise)
    return X, Y

def generate_circles2D_dataset(n_samples=100,noise = 0, factor=0.5):
    X, Y = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    return X, Y
def plot_cluster(X, y):
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=map_to_plt_colors(y))
    plt.show()

#kmeans n_clusters = 3-9

def map_to_plt_colors(y):
    colors_list = list(colors._colors_full_map.values())
    color_pairs = dict(zip(set(y),colors_list))
    mapped_colors = [color_pairs[val] for val in y]
    return mapped_colors

def apply_to_each(item, function):
    #apply function for each item of list of unknown level of nesting
    if isinstance(item,list):
        return [apply_to_each(x, function) for x in item]
    else:
        return function(item)

def add_noise(X,noise_percent):
    return apply_to_each(X, lambda x: np.random.normal(x,np.sqrt(np.abs(x)*noise_percent))) #np.array([[np.random.normal(x,np.sqrt(x*noise_percent)) for x in tpl] for tpl in X])




if __name__ == '__main__':
    #X, y = generate_convex2D_dataset(n_blobs=2, n_samples_per_blob=100, std=[0.2,0.2])
    #X = add_noise(X,noise_percent=0.9)
    #plt.scatter(X[:, 0], X[:, 1], c=map_to_plt_colors(y))
    #plt.title(f'Выпуклый датасет с {1} кластерами')
    #plt.show()
    #print(add_noise([1,2,3],noise_percent=0.1))


    #X, y = generate_convex2D_dataset(n_samples=100, n_features=2)
    n_clusters = 7
    X, y = generate_convex2D_dataset(n_blobs=n_clusters, n_samples_per_blob=100)
    #X, y = load_dataset_laptops()

    plt.figure(figsize=(10, 8))
    plt.scatter(X[:,0], X[:, 1], c=map_to_plt_colors(y))
    plt.title(f'Выпуклый датасет с {n_clusters} кластерами')
    plt.show()

    #1 задача
    """
    for n_clusters in range(2,10):
        kms = KMeans(n_clusters=n_clusters).fit_predict(X)
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=kms, s=10)
        plt.title(f'Число кластеров: {n_clusters}', fontsize=20)
        plt.show()
    """

    #2 задача
    """
    noise_percents = [0.01, 0.03, 0.05, 0.10]
    for noise_percent in noise_percents:

        X_n = add_noise(X, noise_percent)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_n[:, 0], X_n[:, 1], c=map_to_plt_colors(y))
        plt.title(f'Датасет с уровнем шума {noise_percent*100}%')
        plt.show()

        for n_clusters in range(2,10):
            kms = KMeans(n_clusters=n_clusters).fit_predict(X_n)
            fig = plt.figure(figsize=(10, 8))
            plt.scatter(X_n[:, 0], X_n[:, 1], c=kms, s=10)
            plt.title(f'Число кластеров: {n_clusters}; Уровень шума: {noise_percent*100}%', fontsize=20)
            plt.show()
    """
    noise_percents = [0.01, 0.03, 0.05, 0.10]
    #3 задача
    for noise in noise_percents:
        X_c, Y_c = generate_moons2D_dataset(n_samples=300, noise=noise)
        #X_c, Y_c = generate_circles2D_dataset(n_samples=200, noise=0.08, factor=0.5)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_c[:, 0], X_c[:, 1], c=map_to_plt_colors(Y_c))
        plt.title(f'Невыпуклый датасет с 2 кластерами и уровнем шума {noise*100}%')
        plt.show()
        for n_clusters in range(2, 6):
            kms = KMedoids(n_clusters=n_clusters).fit_predict(X_c)
            fig = plt.figure(figsize=(10, 8))
            plt.scatter(X_c[:, 0], X_c[:, 1], c=kms)
            plt.title(f'Число кластеров: {n_clusters}; уровень шума: {noise*100}%', fontsize=20)
            plt.show()



    """
    Внесите шум в набор данных 
    (случайным образом изменить определенную долю объектов набора: 1%, 3%, 5%, 10%; 
    изменение может заключаться в добавлении/вычитании к/из одной/нескольких координат объекта случайного числа).
    Выполните кластеризацию зашумленного набора данных с помощью алгоритмов k‑Means и k‑Medoids (или PAM), 
    используя различные значения параметра  (из интервала 3..9). 
    Выполните визуализацию полученных результатов в виде точечных графиков, на которых цвет точки отражает принадлежность кластеру.
    """

    """
    Выполните кластеризацию набора данных из задания о плотностной кластеризации 
    (с невыпуклыми кластерами) с помощью алгоритмов k‑Means и k‑Medoids (или PAM), 
    используя различные значения параметра  (из интервала 3..9).
    Выполните визуализацию полученных результатов в виде точечных графиков, на которых цвет точки отражает принадлежность кластеру.
    
    """


    """
    X, Y = load_dataset_laptops()

    print(len(set(Y['brand'].tolist())))



    colors_list = list(colors._colors_full_map.values())
    brand_color_pairs = dict(zip(set(Y['brand'].tolist()),colors_list))

    brand_colors = [brand_color_pairs[val] for val in Y['brand']]
    print(brand_colors)

    #colors =

    fig = plt.figure(figsize=(10, 8))
    plt.scatter(X['Price'],X['Rating'],c=brand_colors)
    plt.show()
    """