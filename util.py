import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from pyECLAT import ECLAT

from prettytable import PrettyTable

def visualize_results(items, supports, title, legend, xlabel='', ylabel='', bar=False, grid=False, colors=None,savename=None):

    if colors is None:
        colors = ['r', 'g', 'b', 'y', 'm']
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

    if savename is not None:
        plt.savefig(savename+'.png')

    if grid:
        plt.grid()

    plt.show()
def get_lengths_and_counts(df):
    lengths = df['itemsets'].map(len)
    counts = lengths.value_counts()
    x = counts.index.tolist()
    y = counts.values.tolist()
    x, y = zip(*sorted(zip(x, y)))
    return x, y

def convert_for_pyeclat(dataset):
    max_length = max(len(sublist) for sublist in dataset)
    padded_dataset = [sublist + [None] * (max_length - len(sublist)) for sublist in dataset]
    df = pd.DataFrame(padded_dataset)
    df = df.replace({None: np.NAN})
    return df

def find_association_rules(itemsets, confidence):
    t1 = time.time()
    rules = association_rules(itemsets, metric="confidence", min_threshold=confidence)
    t2 = time.time()
    #print(f'time elapsed:{t2 - t1}')
    return rules, t2-t1

def draw_table_rules(rules, confidence, item_count,max_draw=100,out_name=None):
    rules = rules[rules.apply(lambda row: len(row['antecedents']) + len(row['consequents']) <= item_count, axis=1)]

    rules.sort_values(by='confidence', ascending=False)
    # table = PrettyTable(['antecedent -> consequent','support','confidence'])
    table = PrettyTable(['antecedent', '==>', 'consequent', 'support', 'confidence'])
    for idx, row in rules[['antecedents', 'consequents', 'support', 'confidence']].iterrows():
        table.add_row(
            [f'{list(row["antecedents"])}', ' ==> ', f'{list(row["consequents"])}', row['support'], row['confidence']])
    #    table.add_row([f'{list(row["antecedents"])} ==> {list(row["consequents"])}',row['support'], row['confidence']])
    print(f'table for confidence {confidence}')

    print(f'table {out_name}')
    print(table.get_string(start=0, end=max_draw))
    if out_name is not None:
        with open(out_name+'.txt','w') as file:
            file.write(table.get_string(start=0, end=max_draw))
    print()


def apriori_analyze(dataset, min_support = 1.0, item_count = 7):
    #print('experiment: apriori')

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    print('started apriori')
    t_1 = time.time()
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, low_memory=True)
    t_2 = time.time()
    print(f'time taken: {t_2-t_1} s')

    return frequent_itemsets, t_2-t_1

def eclat_analyze(dataset, min_support = 1.0, item_count = 7):
    df = convert_for_pyeclat(dataset)

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



if __name__ == '__main__':
    # Your dataset
    dataset = [
        ['1', '2', '3'],
        ['1', '2'],
        ['2', '3', '4']
    ]

    # Find the maximum length of any sublist
    max_length = max(len(sublist) for sublist in dataset)

    # Pad each sublist with None up to the maximum length
    padded_dataset = [sublist + [None] * (max_length - len(sublist)) for sublist in dataset]

    # Convert the padded dataset to a DataFrame
    df = pd.DataFrame(padded_dataset)

    # Replace None with NaN for better representation in pandas
    df = df.replace({None: pd.NA})

    print(df)