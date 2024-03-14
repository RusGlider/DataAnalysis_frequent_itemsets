import scipy.cluster.hierarchy as shc
from Task8_Clusterization1 import *

if __name__ == '__main__':
    X, Y = generate_moons2D_dataset(n_samples=400, noise=0.1)
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=map_to_plt_colors(Y))
    plt.title(f'Невыпуклый датасет с 2 кластерами и уровнем шума {10}%')
    plt.show()

    # Rmin(U,V)=min  ρ(u,v), где u∈U,v∈V.
    plt.figure(figsize=(15, 10))
    plt.title('single')
    dend = shc.dendrogram(shc.linkage(X, method='single'))
    plt.show()

    # Rmax(U,V)=max ρ(u,v), где u∈U,v∈V.
    plt.figure(figsize=(15, 10))
    plt.title('complete')
    dend = shc.dendrogram(shc.linkage(X, method='complete'))
    plt.show()

    # Ravg(U,V)=1/(|U|⋅|V|) * ∑u∈U∑v∈Vρ(u,v)
    plt.figure(figsize=(15, 10))
    plt.title('average')
    dend = shc.dendrogram(shc.linkage(X, method='average'))
    plt.show()

    # Rward(U,V)=(|U|⋅|V|)/(|U|+|V|) *ρ2(∑u∈Uu|U|,∑v∈Vv|V|)
    plt.figure(figsize=(15, 10))
    plt.title('ward')
    dend = shc.dendrogram(shc.linkage(X, method='ward'))
    plt.show()




