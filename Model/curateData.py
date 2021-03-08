import pandas as pd
import numpy as np

def expand_tfidf(df):
    df['TF-IDF'] = df['TF-IDF'].apply(lambda x:x.split(' '))
    tf_idf_feature = len(df['TF-IDF'][0])
    features = np.zeros((tf_idf_feature,len(df)))

    for i in range(len(df)):
        for j in range(tf_idf_feature):
            features[j][i] = df['TF-IDF'][i][j]
    for i in range(tf_idf_feature):
        df['feature_' + str(i)] = features[i]
    return df

def curateData(pth, price_col, date_col, n_steps):
    """Reads the dataset and based on n_steps/lags to consider in the time series, creates input output pairs
    Args:
        pth ([str]): [Path to the csv file]
        price_col ([str]): [The name of column in the dataframe that holds the closing price for the stock]
        date_col ([str]): [The nameo oc column in the dataframe which holds dates values]
        n_steps ([int]): [Number of steps/ lags based on which prediction is made]
    """
    df = pd.read_csv(pth,thousands=",").drop(["Unnamed: 0"],axis=1)
    
    df = expand_tfidf(df)
    df = df.drop(["TF-IDF"],axis=1)
    
    # Create lags for the feature columns
    feature_columns = df.columns
    feature_columns = feature_columns.drop(["date","Gold Futures"])

    for col in feature_columns:
        df[col] = df[col].shift()
        
    # Create a dataframe which has only the lags and the date
    new_df = df[[date_col, price_col] + [*feature_columns]]
    new_df = new_df.iloc[n_steps:-1, :]
    new_df = df.iloc[n_steps:-1, :]
    
    # Get a list of dates for which these inputs and outputs are
    dates = list(new_df[date_col])

    # Create input and output pairs out of this new_df
    ips = []
    ops = []
    for entry in new_df.itertuples():
        ip = entry[2:]
        op = entry[6]
        ips.append(ip)
        ops.append(op)

    return (ips, ops, dates)