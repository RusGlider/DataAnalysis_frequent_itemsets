#from apyori import apriori
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
#import pyfpgrowth
from pyECLAT import ECLAT
from pyECLAT import Example1, Example2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import util
#dataset: https://www.kaggle.com/datasets/aslanahmedov/market-basket-analysis/data



def prepare_dataset():
    # df = load_dataset(r'retail dataset/Assignment-1_Data.csv')
    # print(df.head())
    # preprocessing data
    df = pd.read_csv(r'retail dataset/Assignment-1_Data.csv', on_bad_lines='skip', delimiter=';')
    print(df.head())
    print(df.info())

    print("Missing Values:")
    print(df.isnull().sum())

    df.dropna(inplace=True)

    # converting dataframe into transaction data
    transaction_data = df.groupby(['BillNo', 'Date'])['Itemname'].apply(lambda x: ', '.join(x)).reset_index()
    columns_to_drop = ['BillNo', 'Date']
    transaction_data.drop(columns=columns_to_drop, inplace=True)
    transaction_data.to_csv('transaction_data.csv', index=False)

    print("\nTransaction Data for Association Rule Mining:")
    print(transaction_data.head())
    print(transaction_data.shape)

    # Formatting the transaction data in a suitable format for analysis
    # Split the 'Itemname' column into individual items
    items_df = transaction_data['Itemname'].str.split(', ', expand=True)
    print(items_df.head())
    # Concatenate the original DataFrame with the new items DataFrame
    transaction_data = pd.concat([transaction_data, items_df], axis=1)
    # Drop the original 'Itemname' column
    transaction_data = transaction_data.drop('Itemname', axis=1)
    print(transaction_data.head())

    # Convert items to boolean columns
    # df_encoded = pd.get_dummies(transaction_data, prefix='', prefix_sep='').groupby(level=0, axis=1).max()

    # df_encoded.to_csv('transaction_data_encoded.csv', index=False)
    ##df_encoded = pd.read_csv('transaction_data_encoded.csv')

    # df_encoded_part = df_encoded.to_numpy()
    # print('numpy:')
    # print(df_encoded_part)

    # frequent_itemsets = apriori(df_encoded,min_support=0.007)
    # print('frequent itemsets')
    # print(list(frequent_itemsets)[:10])


def apriori_analyze(dataset, min_support = 1.0, item_count = 7):
    print('experiment: apriori')
    #df = pd.read_csv('transaction_data.csv')
    #data = df.to_numpy()
    #new_data = np.array([item.split(',') for item in sublist][0] for sublist in data)

    #df_encoded = pd.read_csv('transaction_data_encoded.csv')
    #df_encoded_part = df_encoded.iloc[:2000]
    #print(df_encoded_part)

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    print('started apriori')
    t_1 = time.time()
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    t_2 = time.time()
    print(f'time taken: {t_2-t_1} s')
    #print(frequent_itemsets)

    #rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    #print("Association Rules:")
    #print(rules)
    #print('ended apriori')
    return frequent_itemsets, t_2-t_1

def eclat_analyze(dataset, min_support = 1.0, item_count = 7):
    df = util.convert_for_pyeclat(dataset)

    t_1 = time.time()
    eclat = ECLAT(data=df)#fpgrowth(df, min_support=0.6, use_colnames=True)
    rule_indices, rule_supports = eclat.fit(min_support=min_support,verbose=False)

    t_2 = time.time()
    print(f'time taken: {t_2 - t_1} s')


    #convert eclat dict into dataframe
    frequent_itemsets = pd.DataFrame(list(rule_supports.items()), columns=['itemsets', 'support'])
        # Split the 'itemsets' column into a list of strings
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].str.split(' & ')
    frequent_itemsets = frequent_itemsets[['support','itemsets']]
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(frozenset)

    return frequent_itemsets, t_2 - t_1

def fpgrowth_analyze(dataset, min_support = 1.0, item_count = 7):
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    print('started fpgrowth')
    t_1 = time.time()
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    t_2 = time.time()
    print(f'time taken: {t_2 - t_1} s')

    return frequent_itemsets, t_2 - t_1


def visualize_results(items, supports, title, legend, xlabel='', ylabel='', bar=False, colors=None):


    if colors is None:
        colors = ['r', 'g', 'b']
    for idx, item in enumerate(items):
        if bar:
            plt.bar(supports,item,color=colors[idx])
        else:
            plt.plot(supports,item,colors[idx])

    plt.xticks(supports)
    plt.legend(legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def get_lengths_and_counts(df):

    lengths = df['itemsets'].map(len)
    counts = lengths.value_counts()
    x = counts.index.tolist()
    y = counts.values.tolist()
    x, y = zip(*sorted(zip(x, y)))
    return x, y



if __name__ == '__main__':

    ex1 = Example1().get()
    print(ex1)

    dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
               ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
               ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']
               ]

    #apriori_analyze()

    # Варьируйте пороговое значение поддержки (например: 1%, 3%, 5%, 10%, 15%, 20%)
    #apriori
    #eclat
    #fpgrowth
    #Подготовьте список частых наборов, в которых не более семи объектов (разумное количество).
    #Визуализация на каждом наборе поддержки




    #количество частых наборов объектов различной длины на фиксированном наборе данных при изменяемом пороге поддержки.
    # вернуть
    item_count = 7
    min_supports = [0.01,0.03,0.05,0.10,0.15,0.20]

    apriori_itemsets_collection = []
    eclat_itemsets_collection = []
    fpgrowth_itemsets_collection = []

    times_apriori = []
    times_eclat = []
    times_fpgrowth = []
    for min_support in min_supports:
        print(f'experiment for support {min_support}')
        itemsets_apriori, time_apriori = apriori_analyze(dataset, min_support, item_count)
        itemsets_eclat, time_eclat = eclat_analyze(dataset, min_support, item_count)
        itemsets_fpgrowth, time_fpgrowth = fpgrowth_analyze(dataset, min_support, item_count)

        apriori_itemsets_collection.append(itemsets_apriori)
        eclat_itemsets_collection.append(itemsets_eclat)
        fpgrowth_itemsets_collection.append(itemsets_fpgrowth)

        times_apriori.append(time_apriori)
        times_eclat.append(time_eclat)
        times_fpgrowth.append(time_fpgrowth)

        #visualize_items_count_per_support
        #visualize_maxlen_per_support
        #visualize_items_count_of_different_lengths_per_support

    #сравнение быстродействия алгоритмов на фиксированном наборе данных при изменяемом пороге поддержки;
    visualize_results(
        items=[times_apriori,times_eclat,times_fpgrowth],
        supports=min_supports,
        title='Быстродействие алгоритмов при разной поддержке',
        legend=['apriori','eclat','fpgrowth'],
        xlabel='Поддержка',
        ylabel='Быстродействие, с'
        )

    # общее количество частых наборов объектов на фиксированном наборе данных при изменяемом пороге поддержки;
    counts_apriori = [len(item) for item in apriori_itemsets_collection]
    counts_eclat = [len(item) for item in eclat_itemsets_collection]
    counts_fpgrowth = [len(item) for item in fpgrowth_itemsets_collection]
    visualize_results(
        items=[counts_apriori, counts_eclat, counts_fpgrowth],
        supports=min_supports,
        title='Количество частых наборов при разной поддержке',
        legend=['apriori', 'eclat', 'fpgrowth'],
        xlabel='Поддержка',
        ylabel='Количество'
    )


    # максимальная длина частого набора объектов на фиксированном наборе данных при изменяемом пороге поддержки;
    itemlen_apriori = [item['itemsets'].apply(len).max() for item in apriori_itemsets_collection]
    itemlen_eclat = [item['itemsets'].apply(len).max() for item in eclat_itemsets_collection]
    itemlen_fpgrowth = [item['itemsets'].apply(len).max() for item in fpgrowth_itemsets_collection]
    visualize_results(
        items=[itemlen_apriori, itemlen_eclat, itemlen_fpgrowth],
        supports=min_supports,
        title='Макс. длина частого набора при разной поддержке',
        legend=['apriori', 'eclat', 'fpgrowth'],
        xlabel='Поддержка',
        ylabel='Длина'
    )

    # количество частых наборов объектов различной длины на фиксированном наборе данных при изменяемом пороге поддержки.
    for idx, min_support in enumerate(min_supports):
        x_apriori, y_apriori = get_lengths_and_counts(apriori_itemsets_collection[idx])
        x_eclat, y_eclat = get_lengths_and_counts(eclat_itemsets_collection[idx])
        x_fpgrowth, y_fpgrowth = get_lengths_and_counts(fpgrowth_itemsets_collection[idx])


        visualize_results(
            items=[y_apriori],
            supports=x_apriori,
            title=f'Количество частых наборов различной длины при поддержке {min_support}',
            legend=['apriori'],
            xlabel='Длина',
            ylabel='Количество',
            bar=True,
            colors = ['r']
        )
        visualize_results(
            items=[y_eclat],
            supports=x_eclat,
            title=f'Количество частых наборов различной длины при поддержке {min_support}',
            legend=['eclat'],
            xlabel='Длина',
            ylabel='Количество',
            bar=True,
            colors = ['g']
        )
        visualize_results(
            items=[y_fpgrowth],
            supports=x_fpgrowth,
            title=f'Количество частых наборов различной длины при поддержке {min_support}',
            legend=['fpgrowth'],
            xlabel='Длина',
            ylabel='Количество',
            bar=True,
            colors = ['b']
        )






    """
    df = pd.read_csv('transaction_data.csv')
    data = df.to_numpy()
    new_data = list([item.split(',') for item in sublist][0] for sublist in data)
    part_data = new_data[:10]
    """
    """
    
    """

    """
    te = TransactionEncoder()
    te_ary = te.fit(part_data).transform(part_data)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    print('started apriori')
    t_1 = time.time()
    frequent_itemsets = apriori(df, min_support=0.1,use_colnames=True)
    t_2 = time.time()
    print(f'time taken: {t_2 - t_1} s')
    print(frequent_itemsets)
    """

    #prepare_dataset()
    #df_encoded = pd.read_csv('transaction_data_encoded.csv')
    #frequent_itemsets = apriori(df_encoded, min_support=0.007)
    #rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    #print("Association Rules:")
    #print(rules.head())


