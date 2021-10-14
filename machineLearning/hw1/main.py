import functools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def scale_decorator(func):
    """Scale decorator.
    
    Keyword arguments:
    func -- scale function to decorate

    """
    @functools.wraps(func)
    def wrapper(features: pd.DataFrame) -> pd.DataFrame:
        features = func(features).dropna(axis=1, how='all')
        #features.insert(0, 'ones', np.arange(len(features.index)))
        features.columns = np.arange(len(features.columns))
        return features
    return wrapper

@scale_decorator
def standardize(features: pd.DataFrame) -> pd.DataFrame:
    """Scale by standardization.
    
    Keyword arguments:
    features -- features table

    """
    features = (features - features.mean()) / features.var()
    return features

@scale_decorator
def segment(features: pd.DataFrame) -> pd.DataFrame:
    """Scale by segment.
    
    Keyword arguments:
    features -- features table

    """
    features = (features - features.min()) / (features.max() - features.min())
    return features


def gradient(features: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Gradient of MSE.
        
    Keyword arguments:
    features -- features table
    w        -- current scales vector

    """
    length = len(features.columns) - 1
    x = features[np.arange(length)]
    trps = x.transpose()
    return trps.dot(x.dot(w) - features[length])

def learn(features: pd.DataFrame, epsilon=0.000001) -> pd.Series:
    """Learn by Gradient descent.
        
    Keyword arguments:
    features -- features table

    """
    w = pd.Series(np.zeros(len(features.columns) - 1))
    grad = gradient(features, w)
    k = 1
    while abs(grad.sum()) >= epsilon:
        w -= grad / k
        grad = gradient(features, w)
        k += 1
    return w


def root_mean_squared_error(features: pd.DataFrame, w: pd.Series):
    """RMSE deflection function.
        
    Keyword arguments:
    features -- features table
    w        -- current scales vector

    """
    length = len(features.columns) - 1
    y = features[np.arange(length)].dot(w)
    return mean_squared_error(features[length], y, squared=False)

def determination(features: pd.DataFrame, w: pd.Series):
    """Coefficient of determination (R^2).
        
    Keyword arguments:
    features -- features table
    w        -- current scales vector

    """
    length = len(features.columns) - 1
    y = features[np.arange(length)].dot(w)
    return r2_score(features[length], y)


test_1 = segment(pd.read_csv('./Dataset/Features_Variant_1.csv'))
test_2 = segment(pd.read_csv('./Dataset/Features_Variant_2.csv'))
test_3 = segment(pd.read_csv('./Dataset/Features_Variant_3.csv'))
test_4 = segment(pd.read_csv('./Dataset/Features_Variant_4.csv'))
test_5 = segment(pd.read_csv('./Dataset/Features_Variant_5.csv'))

train_1 = pd.concat([test_2, test_3, test_4, test_5])
train_2 = pd.concat([test_1, test_3, test_4, test_5])
train_3 = pd.concat([test_1, test_2, test_4, test_5])
train_4 = pd.concat([test_1, test_2, test_3, test_5])
train_5 = pd.concat([test_1, test_2, test_3, test_4])

w_1 = learn(train_1)
w_2 = learn(train_2)
w_3 = learn(train_3)
w_4 = learn(train_4)
w_5 = learn(train_5)

w_1.to_csv('./Result/W_1.csv')
w_2.to_csv('./Result/W_2.csv')
w_3.to_csv('./Result/W_3.csv')
w_4.to_csv('./Result/W_4.csv')
w_5.to_csv('./Result/W_5.csv')


test_rmse = [root_mean_squared_error(test_1, w_1), root_mean_squared_error(test_2, w_2),
             root_mean_squared_error(test_3, w_3), root_mean_squared_error(test_4, w_4),
             root_mean_squared_error(test_5, w_5)]

train_rmse = [root_mean_squared_error(train_1, w_1), root_mean_squared_error(train_2, w_2),
              root_mean_squared_error(train_3, w_3), root_mean_squared_error(train_4, w_4),
              root_mean_squared_error(train_5, w_5)]

test_r2 = [determination(test_1, w_1), determination(test_2, w_2),
           determination(test_3, w_3), determination(test_4, w_4),
           determination(test_5, w_5)]

train_r2 = [determination(train_1, w_1), determination(train_2, w_2),
            determination(train_3, w_3), determination(train_4, w_4),
            determination(train_5, w_5)]


test_table = pd.DataFrame(np.array([test_rmse, test_r2]), columns=['T1', 'T2', 'T3', 'T4', 'T5'])
test_table.insert(5, 'E', test_table['T1', 'T2', 'T3', 'T4', 'T5'].mean())
test_table.insert(6, 'STD', test_table['T1', 'T2', 'T3', 'T4', 'T5'].var())
test_table.insert(0, 'Type', ['RMSE', 'R^2'])
test_table.set_index('Type')

train_table = pd.DataFrame(np.array([train_rmse, train_r2]), columns=['T1', 'T2', 'T3', 'T4', 'T5'])
train_table.insert(5, 'E', train_table['T1', 'T2', 'T3', 'T4', 'T5'].mean())
train_table.insert(6, 'STD', train_table['T1', 'T2', 'T3', 'T4', 'T5'].var())
train_table.insert(0, 'Type', ['RMSE', 'R^2'])
train_table.set_index('Type')

print(test_table)
print(train_table)
