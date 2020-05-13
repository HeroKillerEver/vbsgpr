import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.cluster.vq import kmeans2 as kmeans
import warnings
warnings.filterwarnings("ignore")


def load(file, n_clusters=1000, n_induce=100):
    
    df = pd.read_csv('./data/{}'.format(file))
    N, D = df.shape
    cols = [ chr(ord('a') + i) for i in range(D)]
    print ('loading data of {} datapoints...'.format(N))
    df = df.values
    df = pd.DataFrame(df, columns=cols)

    print ('Parition into {} clusters...'.format(n_clusters))
    _, clusters = kmeans(df.values.astype('float32'), n_clusters, minit='points')
    print ('Done...!')
    df['cluster'] = pd.Series(clusters, index=df.index)

    # X = df_airline.iloc[:, :-2]
    # y = df_airline.iloc[:, -2:]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=2018)

    df_train, df_test = train_test_split(df, test_size=0.01, random_state=2018)


    normalizer = StandardScaler()

    # X_train_norm = normalizer_x.fit_transform(X_train)
    # y_train_norm = normalizer_y.fit_transform(y_train.values[:, None])
    # X_test_norm = normalizer_x.fit_transform(X_test)
    # y_test_norm = normalizer_y.fit_transform(y_test.values[:, None])

    df_train_norm = normalizer.fit_transform(df_train.values[:, :-1]) # no cluster , has label

    print ('Selecting {} inducing variables...'.format(n_induce))
    z, _ = kmeans(df_train_norm[:, :-1], n_induce, minit='points')
    print ('Done...!')

    df_test_norm = normalizer.fit_transform(df_test.values[:, :-1])  # no cluster

    data_train_norm = np.column_stack((df_train_norm, df_train.cluster.values)) # with cluster
    data_test_norm = np.column_stack((df_test_norm, df_test.cluster.values)) # with cluster

    y_std = normalizer.scale_[-1]

    return data_train_norm, data_test_norm, z, y_std


    





if __name__ == '__main__':

    _, _, _, std = load('airplane.csv')
