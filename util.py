import pandas as pd
import numpy as np

def convert_for_pyeclat(dataset):
    max_length = max(len(sublist) for sublist in dataset)
    padded_dataset = [sublist + [None] * (max_length - len(sublist)) for sublist in dataset]
    df = pd.DataFrame(padded_dataset)
    df = df.replace({None: np.NAN})
    return df

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