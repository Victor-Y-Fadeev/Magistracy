import functools
import math
import numpy as np
import pandas as pd


def scale_decorator(func):
    """Scale decorator.
    
    Keyword arguments:
    func -- scale function to decorate

    """
    @functools.wraps(func)
    def wrapper(features: pd.DataFrame) -> pd.DataFrame:
        features = func(features).dropna(axis=1, how='all')
        features.insert(0, 'ones', np.arange(len(features.index)))
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
    w_temp = x.transpose().dot(x.dot(w) - features[length])
    return 2 * w_temp / len(x.columns)

def learn(features: pd.DataFrame, epsilon=0.00001) -> pd.Series:
    """Learn by Gradient descent.
        
    Keyword arguments:
    features -- features table

    """
    w = pd.Series(np.ones(len(features.columns) - 1))
    grad = gradient(features, w)
    i = 1000
    while abs(grad.sum()) >= epsilon:
        w -= grad / i
        grad = gradient(features, w)
        i += 1
    return w


def root_mean_squared_error(features: pd.DataFrame, w: pd.Series):
    """RMSE deflection function.
        
    Keyword arguments:
    features -- features table
    w        -- current scales vector

    """
    length = len(features.columns) - 1
    y = features[np.arange(length)].dot(w)
    diff = y - features[length]
    return math.sqrt(diff.pow(2).sum() / len(features.index))

def determination(features: pd.DataFrame, w: pd.Series):
    length = len(features.columns) - 1
    y = features[np.arange(length)].dot(w)
    return 1 - (y - features[length]).pow(2).sum() / (y - features[length].mean()).pow(2).sum()



features = pd.read_csv('./Dataset/Features_Variant_5.csv')

features = segment(features)
#print(features[len(features.columns) - 1])

w = learn(features)
w.to_csv('./Features_1.csv')
print(root_mean_squared_error(features, w))
print(determination(features, w))


#print(features[:1].isnull()[37])
#features[:1].to_csv('./Features_1.csv')