#from apyori import apriori
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder


from pyECLAT import ECLAT
from pyECLAT import Example1, Example2


import numpy as np
import pandas as pd
import time

from util import *
import task1
import task2
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



if __name__ == '__main__':
    df = pd.read_csv('transaction_data.csv')
    data = df.to_numpy()
    new_data = list([item.split(',') for item in sublist][0] for sublist in data)

    #df_encoded = pd.read_csv('transaction_data_encoded.csv')
    #df_encoded_part = df_encoded.iloc[:2000]
    #print(df_encoded_part)
    dataset = new_data[:5]
    print(dataset)
    """
    dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
               ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
               ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']
               ]
    """
    #task1(dataset)
    #task2(dataset)