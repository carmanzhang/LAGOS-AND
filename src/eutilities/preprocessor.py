import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def drop_missing_items(df):
    df = df.dropna(how='any')
    return df


def down_sample(df, percent=1):
    '''
    percent:多数类别下采样的数量相对于少数类别样本数量的比例
    '''
    neg_samples_num = df[df['same_author'] == 0].shape[0]
    pos_samples_num = df[df['same_author'] == 1].shape[0]
    if neg_samples_num < pos_samples_num:
        data1 = df[df['same_author'] == 0]  # 将多数类别的样本放在data1
        data0 = df[df['same_author'] == 1]  # 将少数类别的样本放在data0
    else:
        data0 = df[df['same_author'] == 0]  # 将多数类别的样本放在data0
        data1 = df[df['same_author'] == 1]  # 将少数类别的样本放在data1
    index = np.random.randint(
        len(data0), size=percent * (len(df) - len(data0)))  # 随机给定下采样取出样本的序号
    lower_data1 = data0.iloc[list(index)]  # 下采样
    # print(lower_data1.shape)
    # print(data1.shape)
    return (pd.concat([lower_data1, data1]))


def scale(df):
    mm_scaler = MinMaxScaler()
    df = mm_scaler.fit_transform(df)
    std_scaler = StandardScaler()
    df = std_scaler.fit_transform(df)
    return df


def select_features():
    # #SelectKBest（卡方系数）
    # ch2 = SelectKBest(chi2,k=3)#在当前的案例中，使用SelectKBest这个方法从4个原始的特征属性，选择出来3个
    # x_train = ch2.fit_transform(x_train, y_train)#训练并转换
    # select_name_index = ch2.get_support(indices=True)
    # print ("对类别判断影响最大的三个特征属性分布是:",ch2.get_support(indices=False))
    # print(select_name_index)
    pass


def preprocess(df):
    print('original shape: ', df.shape)
    df = drop_missing_items(df)
    print('after dropping shape: ', df.shape)
    df = scale(df)
    print('after scaling shape: ', df.shape)
    df = down_sample(df)
    print('after sampling shape: ', df.shape)
    return df
