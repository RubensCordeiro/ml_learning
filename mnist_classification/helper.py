
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline


def data_loader(folder_path, file_name, dataset_type,  classificator_type='binary', selected_number=6):
    """This function will help to select data based on the type of classification intented"""

    file_path = folder_path + "/" + file_name
    complete_data = pd.read_csv(file_path)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=42)

    split_indices = splitter.split(complete_data, complete_data['label'])

    for train, test in split_indices:
        train_set = complete_data.loc[train]
        test_set = complete_data.loc[test]

    if(dataset_type == 'train'):
        x_data = train_set.drop('label', axis='columns').sort_index()
        y_data = train_set['label'].sort_index()

    else:
        x_data = test_set.drop('label', axis='columns').sort_index()
        y_data = test_set['label'].sort_index()

    if(classificator_type == 'binary'):
        y_data = (y_data == selected_number)
        return (x_data, y_data)

    elif(classificator_type == 'multinominal'):
        return (x_data, y_data)

    elif(classificator_type == 'multilabel'):
        y_data_large = (y_data >= 7)
        y_data_even = (y_data % 2 == 0)
        y_data = np.c_[y_data_large, y_data_even]
        return(x_data, y_data)


class DataSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x_data, y_data=None):
        return self

    def transform(self, x_data, y_data=None):
        return x_data


data_pipeline = Pipeline([
    ('selector', DataSelector()),
    ('scaler', StandardScaler())
])
