import matplotlib.pyplot as plt

from util import *
#from Task12_Anomaly1 import load_dataset_laptops

from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import matrixprofile as mp
def clusterization(df):
    # кластеризация
    fig = plt.figure(figsize=(12, 8))
    plt.xlabel('Цена, $')
    plt.ylabel('Рейтинг')
    plt.title('Рейтинг ноутбука в зависимости от его стоимости')
    plt.scatter(df['Price']/82, df['Rating'], cmap='inferno', s=10)
    plt.show()


    dbclass = DBSCAN(eps=200*82, min_samples=10).fit_predict(df[['Price', 'Rating']])

    plt.xlabel('Цена, $')
    plt.ylabel('Рейтинг')
    plt.title('Аномалии с разрывом более 200 долларов')
    plt.scatter(df['Price']/82, df['Rating'], c=dbclass, s=10)
    plt.show()


def loops(df, r=0.3, a=0.01):

    data = df.values
    is_anomaly = np.zeros(len(data))
    for i, x1 in enumerate(data):
        INOUT = []
        for j, x2 in enumerate(data):
            if i != j and distance.euclidean(data[i], data[j]) <= r:
                INOUT.append(j)
        if len(INOUT) <= a*len(data):
            is_anomaly[i] = 1
            for j, aj in enumerate(INOUT):
                if is_anomaly[aj] != 1:
                    is_anomaly[aj] = 1

    return is_anomaly


def discord_anomalies(X, window_size=19):
    profile = mp.compute(X, window_size)
    profile = mp.discover.discords(profile, k=1, exclusion_zone=window_size)
    mp.visualize(profile)
    plt.show()

    mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(16, 4))
    ax.plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
    ax.set_title('Window Size {}'.format(str(window_size)))
    ax.set_ylabel('Data')
    flag = 1
    for discord in profile['discords']:
        x = np.arange(discord, discord + profile['w'])
        y = profile['data']['ts'][discord:discord + profile['w']]
        if flag:
            ax.plot(x, y, c='r', label="Discord")
            flag = 0
        else:
            ax.plot(x, y, c='r')
    plt.legend()
    plt.show()


# time series dataset https://www.kaggle.com/datasets/jacobsharples/ontario-electricity-demand

if __name__ == '__main__':
    """
    df = pd.read_csv('laptops.csv')

    #clusterization(df)
    
    inp_df = df[['Price','Rating']]
    scaler = MinMaxScaler()
    scaler.fit(inp_df)
    normalized_df = pd.DataFrame(scaler.transform(inp_df), columns=inp_df.columns)
    is_anomaly = loops(normalized_df)
    #OKNOT = anomalies_grid(df['Price'],df['Rating'])



    plt.xlabel('Цена, $')
    plt.ylabel('Рейтинг')
    plt.title('Аномалии методом вложенных циклов')
    plt.scatter(df['Price'] / 82, df['Rating'], c=is_anomaly, s=10)
    plt.show()
    """

    #df = pd.read_csv('weather_history_bangladesh.csv')
    #inp_df = df[['temperature_fahrenheit']]
    #plt.scatter(range(len(inp_df)),inp_df)

    df = pd.read_csv('ontario_electricity_demand.csv')
    #inp_df = df[['hourly_demand']][10000:17000]
    inp_df = df[['hourly_demand']][11000:12000]
    #inp_df = df[['hourly_demand']][12000:13000]

    plt.plot(inp_df/1000)
    plt.ylabel('Потребление, КВт*ч')
    plt.xlabel('Время')
    plt.show()


    data = inp_df.to_numpy()
    datax = [x[0] for x in data]
    #data = [[i,x[0]] for i, x in enumerate(data)]
    discord_anomalies(datax,window_size=12) #150


