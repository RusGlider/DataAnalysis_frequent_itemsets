from Task8_Clusterization1 import *
from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    #X, Y = generate_moons2D_dataset(n_samples=400, noise=0.1)

    epses=[0.004,0.008,0.016,0.032,0.064]
    noise_percents = [0.01, 0.03, 0.09, 0.16]
    min_samples = [2,3,4,5,6,7,8,9]
    #noise_percents = [0.4, 0.4, 0.3, 0.4]
    for n, noise in enumerate(noise_percents):
        #X, Y = generate_moons2D_dataset(n_samples=400, noise=noise)
        X, Y = generate_convex2D_dataset(n_blobs=7, n_samples_per_blob=100)
        X = add_noise(X, noise)
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=map_to_plt_colors(Y))
        plt.title(f'выпуклый датасет с 7 кластерами и уровнем шума {noise * 100}%')
        plt.show()

        fig, axs = plt.subplots(len(min_samples), len(epses),figsize=(15,10))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.7)
        #fig(figsize=(10,10))
        for i, min_sample in enumerate(min_samples):
            #plt.figure(figsize=(15,10))

            for j, eps in enumerate(epses):
                dbclass = DBSCAN(eps=eps, min_samples=min_sample).fit_predict(X)
                #plt.subplot(len(), len(epses), plot_num)
                axs[i,j].scatter(X[:, 0], X[:, 1], c=dbclass, s=8)
                axs[i,j].set_title(f'min_sample:{min_sample}, eps:{eps}')

        plt.show()
