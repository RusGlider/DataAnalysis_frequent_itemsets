import matplotlib.pyplot as plt
import numpy as np

from util import *

from scipy.stats import chi2, chisquare
import math


def load_dataset_laptops():
    # dataset https://www.kaggle.com/datasets/bhavikjikadara/brand-laptops-dataset
    df = pd.read_csv('laptops.csv')
    df.drop(["index"], axis=1, inplace=True)

    # index, brand, Model, Price, Rating, processor_brand, processor_tier, num_cores, num_threads, ram_memory,
    # primary_storage_type, primary_storage_capacity, secondary_storage_type, gpu_brand, gpu_type, is_touch_screen,
    # display_size, resolution_width, resolution_height, OS, year_of_warranty
    X = df[['Price','Rating']].values
    Y = df[['brand']].values
    Y = [item[0] for item in Y]
    return X, Y

def plot_1d(X, Y,title='',xlabel='', ylabel=''):
    fig = plt.figure(figsize=(10,6))
    img = plt.scatter(X, [0]*len(X), c=Y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #img.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(img, shrink=0.8)
    plt.show()


def MLE(X):
    def f(n):
        try:
            return math.lgamma(n)
        except:
            print('bad n', n)
            return 1
    N = np.sum(X)

    theta = X / np.sum(X)
    mle_ = f(N) - np.sum([f(r) for r in X]) + np.sum([r * theta[i] for i, r in enumerate(X)])

    aic = 2 * len(X) - 2 * math.log(mle_)
    return mle_, aic

def max_likelihood_estimation(X):
    #theta = X / np.sum(X)

    theta = X / np.sum(X)
    print('MLE: ', np.round(theta,3))

    def f(n):
        #return math.factorial(n)
        return math.lgamma(n)

    def mle(R, theta):
        return (f(np.sum(R)) / (np.prod([f(r) for r in R]))) * np.prod([theta[i] ** R[i] for i in range(len(R))])
        # максимальное правдоподобие


    ml = mle(X, theta)

    print("ML: {:.7f}".format(ml))

    aic = 2 * len(X) - 2 * math.log(ml)
    print("AIC: {:.3f}".format(aic))

    """
    # количества звезд по отзывам
    R_A = np.array([10, 6, 10, 27, 109])
    R_B = np.array([57, 33, 29, 45, 246])

    # оценки максимального правдоподобия
    theta_A = R_A / np.sum(R_A)
    theta_B = R_B / np.sum(R_B)

    print("MLE for Product A: ", np.round(theta_A, 3))
    print("MLE for Product B: ", np.round(theta_B, 3))

    def f(n):
        return math.lgamma(n)

    def mle(R, theta):
        return (f(np.sum(R)) / (np.prod([f(r) for r in R]))) * np.prod([theta[i] ** R[i] for i in range(len(R))])
        #return (f(np.sum(R)) / (np.prod([f(r) for r in R]))) * np.prod([theta[i] ** R[i] for i in range(len(R))])

    # максимальное правдоподобие
    ml_A = mle(R_A, theta_A)
    ml_B = mle(R_B, theta_B)

    print("ML for Product A {:.7f}".format(ml_A))
    print("ML for Product B {:.7f}".format(ml_B))

    aic_A = 2 * 5 - 2 * math.log(ml_A)
    aic_B = 2 * 5 - 2 * math.log(ml_B)
    print("AIC for Product A: {:.2f}".format(aic_A))
    print("AIC for Product B: {:.2f}".format(aic_B))
    """


def xi_squared(X):
    fig, ax = plt.subplots(1, 1)
    # График частот значений
    ax.hist(df['Price'], bins=30, histtype='stepfilled', orientation='vertical', color='purple')

    x = np.arange(np.min(X), np.max(X), 0.01)

    ch = chi2.pdf(x * 30, df=4) * 1100
    # ch = (ch - min(ch)) / max(ch)
    # Эмпирически подбираем функцию, описывающую распределение
    ax.plot(x, ch, 'r-', lw=5, alpha=0.6, label='chi2 pdf')
    plt.show()

    mu, sigma = 0.1, 0.002  # гипотеза, что распределение является нормальным
    x_expected = np.random.normal(mu, sigma, len(X))
    """
    min_pval = 1.0
    min_statistic = 999999999
    exz_mu = 1.0
    exz_sigma = 1.0
    exz_pres = None
    for mu in np.arange(0.8,0.11,0.001):
        for sigma in np.arange(0.0001,0.005,0.0001):
            x_expected = np.random.normal(mu,sigma,len(X))
            pres = chisquare(f_obs=X, f_exp=( (np.sum(X)/np.sum(x_expected)) * x_expected))
            pval = pres.pvalue
            stat = pres.statistic
            #print(f'mu: {mu}, sigma: {sigma}, pres: {pres}')
            if pval < min_pval and stat < min_statistic:
                min_pval = pval
                min_statistic = stat
                exz_mu = mu
                exz_sigma = sigma
                exz_pres = pres
    if exz_pres is not None:
        print(f'mu: {exz_mu}, sigma: {exz_sigma}, pvalue: {exz_pres.pvalue}, statistic: {exz_pres.statistic}')
    else:
        print('Found none getting the hypothesis')
    """
    print(chisquare(f_obs=X, f_exp=((np.sum(X) / np.sum(x_expected)) * x_expected)))


def anomaly_histogram(X, Y):

    fig = plt.figure(figsize=(15,10))

    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
    ax_1 = fig.add_subplot(grid[:-1, :-1])
    ax_1.scatter(X,Y)
    ax_1.set(title='Рейтинг ноутбука в зависимости от Стоимости', xlabel='Стоимость', ylabel='Рейтинг')

    ax_bottom = fig.add_subplot(grid[-1, 0:-1])
    ax_bottom.hist(X, bins=30, histtype='stepfilled', orientation='vertical', color='blue')

    ax_right = fig.add_subplot(grid[:-1, -1])
    ax_right.hist(Y, bins=30, histtype='stepfilled', orientation='horizontal', color='red')

    plt.show()


def anomaly_maximum_likelihood(df):
    X_Ratings = np.array([len(df['Price'][df['Rating'] == R]) for R in range(min(df['Rating']), max(df['Rating']) + 1)])
    X_Ratings_norm = (X_Ratings - min(X_Ratings)) / max(X_Ratings)
    plt.bar(range(min(df['Rating']), max(df['Rating']) + 1), X_Ratings, alpha=0.5, color='g')
    plt.title('Агрегация ноутбуков по рейтингу')
    plt.xlabel('Рейтинг')
    plt.ylabel('Количество')
    # plt.show()

    # гипотеза, что распределение рейтингов является нормальным
    x_expected = np.random.normal(loc=65, scale=9, size=np.sum(X_Ratings))

    x_expected = x_expected[(x_expected >= min(df['Rating'])) & (x_expected <= max(df['Rating']))]
    #plt.bar(range(min(df['Rating']), max(df['Rating']) + 1), x_expected_bins, alpha=0.5, color='r')
    plt.hist(x_expected, bins=len(X_Ratings), density=False, alpha=0.5, color='r')
    plt.title('Столбчатая диаграмма реальных значений и нормального распределения')
    plt.xlabel('Рейтинг')
    plt.ylabel('Кол-во вхождений')
    plt.legend(['real','expected'])
    # plt.bar(range(min(x_expected),max(x_expected)+1),x_expected)
    # plt.bar(range(min(x_expected),max(x_expected)+1),int(x_expected))
    plt.show()

    x_expected_bins, _ = np.histogram(x_expected,bins=len(X_Ratings))
    ml_real, aic_real = MLE(X_Ratings)
    ml_expect, aic_expect = MLE(x_expected_bins)

    print("ML expected: {:.7f}".format(ml_expect))
    print("ML real: {:.7f}".format(ml_real))

    print("AIC expected: {:.3f}".format(aic_expect)) # Информационный критерий Акаике
    print("AIC real: {:.3f}".format(aic_real))
    """
    Оценка по информационному критерию Акаике награждает модели, которые 
    достигают высокого показателя критерия адекватности (с низким максимальным правдоподобием), 
    и штрафует модели, если они становятся чрезмерно сложными (большое количество параметров k)
    """

    # гипотеза о
    # max_likelihood_estimation(X_Ratings)



if __name__ == '__main__':
    #X, Y = load_dataset_laptops()
    df = pd.read_csv('laptops.csv')
    #plot_1d(df['Price'], df['Rating'],title='Стоимость ноутбуков',xlabel='Стоимость', )
    #plot_1d(df['num_cores'], df['Rating'])

    plt.scatter(df['Price'],df['Rating'])
    plt.title('Рейтинг ноутбука в зависимости от Стоимости')
    plt.xlabel('Стоимость')
    plt.ylabel('Рейтинг')
    plt.show()

    anomaly_histogram(df['Price'], df['Rating'])

    anomaly_maximum_likelihood(df)









