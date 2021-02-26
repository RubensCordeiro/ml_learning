
"Hyperparemeter class inpired BY: http://www.davidsbatista.net/blog/2018/02/23/model_optimization/"

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def data_loader(folder_path, file_name, dataset_type,  classificator_type='binary', selected_number=6):
    """This function will help to select data based on the type of classification intented"""

    file_path = folder_path + "/" + file_name
    complete_data = pd.read_csv(file_path)
    complete_data = complete_data.groupby('label', as_index=False)
    complete_data = complete_data.apply(lambda x: x.sample(n=500))
    complete_data = complete_data.reset_index(drop=True)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=42)

    split_indices = splitter.split(complete_data, complete_data['label'])

    for train, test in split_indices:
        train_set = complete_data.loc[train].copy()
        test_set = complete_data.loc[test].copy()

    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    if(dataset_type == 'train'):
        x_data = train_set.drop('label', axis='columns').copy()
        y_data = train_set['label'].reset_index(drop=True).copy()

    else:
        x_data = test_set.drop('label', axis='columns').copy()
        y_data = test_set['label'].reset_index(drop=True).copy()

    if(classificator_type == 'binary'):
        y_data = (y_data == selected_number)
        return (x_data, y_data)

    elif(classificator_type == 'multiclass'):
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
    ('scaler', MinMaxScaler())
])


class multiModelSelection():
    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.model_names = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, **grid_kwargs):
        for name in self.model_names:
            print(f'Gridsearch for {name} model')
            model = self.models[name]
            params = self.params[name]
            grid_search = GridSearchCV(model, params, **grid_kwargs)
            grid_search.fit(X, y)
            self.grid_searches[name] = grid_search
        print('Done.')

    def display_results(self, sort_by='mean_test_score'):
        df_rows = []
        for name, grid_search in self.grid_searches.items():
            df_row = pd.DataFrame(grid_search.cv_results_)
            removed_columns = [
                col for col in df_row.columns if 'params_' not in col]
            df_row = df_row.filter(removed_columns, axis='columns')
            df_row['Chosen Model'] = len(df_row)*[name]
            df_rows.append(df_row)
        final_df = pd.concat(df_rows)

        final_df = final_df.sort_values([sort_by], ascending=False)
        final_df = final_df.reset_index()
        final_df = final_df.drop(['rank_test_score', 'index'], 1)

        columns = final_df.columns.tolist()
        columns.remove('Chosen Model')
        columns = ['Chosen Model'] + columns
        final_df = final_df[columns]
        return final_df


class classificationEval():
    def __init__(self, y_true, y_pred, y_probs):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_probs = y_probs

    def confusion_matrix(self):
        return ConfusionMatrixDisplay(confusion_matrix(self.y_true, self.y_pred)).plot()

    def main_metrics(self, **evalkwargs):
        precision = precision_score(
            self.y_true, self.y_pred, **evalkwargs).round(2)

        recall = recall_score(self.y_true, self.y_pred,
                              **evalkwargs).round(2)

        f1 = f1_score(self.y_true, self.y_pred,  **evalkwargs).round(2)
        roc = roc_auc_score(self.y_true, self.y_pred, **evalkwargs).round(2)

        df = pd.DataFrame([[precision, recall, f1, roc]],
                          columns=['precision', 'recall', 'F1', 'ROC AUC'])

        return df

    def roc_curves(self, inverted=False):

        if(not inverted):
            fp, tp, th = roc_curve(self.y_true, self.y_probs)
            plt.figure(figsize=(15, 10))
            plt.plot(tp, fp, label='label')
            plt.plot([0, 1], [0, 1])
            plt.axis([-0.05, 1.05, -0.05, 1.05])
            plt.xlabel('False Positives')
            plt.ylabel('True Positives')
        else:
            fp, tp, th = roc_curve(self.y_true, self.y_probs)
            plt.figure(figsize=(15, 10))
            plt.plot(fp, tp, label='label')
            plt.plot([0, 1], [0, 1])
            plt.axis([-0.05, 1.05, -0.05, 1.05])
            plt.xlabel('False Positives')
            plt.ylabel('True Positives')
