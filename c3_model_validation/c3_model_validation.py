import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.neighbors import LocalOutlierFactor
import xgboost as xgb

style.use('ggplot')


def regression_precision_error(y_true, y_pred):
    """Calculate absolute average error."""
    return np.sum(np.absolute(y_true - y_pred) / y_true) / len(y_true)


def validation(models, model_names, random_state=2):
    """
    Given a set of models, and the list of their corresponding name, process the data,
    and then fit the data into each model, validate the model performance by train-test-split with 8:2 ratio with MAE
    and AAD metrics, save the scores into a dataframe.

    return pd.DataFrame
    """
    drop_cols = ['token',
                 'serial_number',
                 'model',
                 'package_name',
                 'start_time',
                 'exit_time',
                 'min_ambient_temp',
                 'max_ambient_temp',
                 '2_8_range_duration']

    target_name = '2_8_range_duration'

    # Load json
    df = pd.read_json('../data/c3_json.json', orient='records')

    # Engineer features
    df['max_to_min_ratio'] = df['max_ambient_temp'] / df['min_ambient_temp']
    # df['min_to_max_ratio'] = df['min_ambient_temp'] / df['max_ambient_temp']

    # Remove outliers
    dff = df[['min_ambient_temp', 'max_ambient_temp', 'ambient_MKT_value']].copy()

    lof = LocalOutlierFactor()

    yhat = lof.fit_predict(dff)

    mask = yhat != -1

    new_df = df.values[mask, :]
    new_df = pd.DataFrame(new_df, columns=df.columns)

    train_df = new_df.drop(drop_cols, axis=1).copy()
    target_var = new_df[target_name].values

    # Cross validation
    X_train, X_test, y_train, y_test = train_test_split(train_df, target_var,
                                                        test_size=0.2,
                                                        random_state=random_state)
    scores = []

    for model in models:
        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

        # A pipeline to process the feature data
        pipeline = Pipeline(steps=[('knn', imputer),
                                   ('m', model)])

        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        scores.append({
            'train_MAE': mean_absolute_error(y_train, train_pred),
            'test_MAE': mean_absolute_error(y_test, test_pred),
            'train_AAD': regression_precision_error(y_train, train_pred),
            'test_AAD': regression_precision_error(y_test, test_pred)
        })

    return pd.DataFrame(scores, index=model_names)


def feature_importance_plot(model, scoring='neg_mean_absolute_error', plot=False, save_plot=False):
    """
    Plot feature importance based on permutation importance.
    :param model: pre-trained ml model
    :param scoring: (str, default neg_mean_absolute_error)
    :param plot: (bool, default False) if True, plot feature importance plot.
    :param save_plot: (bool, default False) if True, save the feature importance plot as png in the same directory.
    :return: None
    """
    drop_cols = ['token',
                 'serial_number',
                 'model',
                 'package_name',
                 'start_time',
                 'exit_time',
                 'min_ambient_temp',
                 '2_8_range_duration',
                 'max_ambient_temp']

    target_name = '2_8_range_duration'

    # Load json
    df = pd.read_json('../data/c3_json.json', orient='records')

    # Engineer features
    df['max_to_min_ambient_ratio'] = df['max_ambient_temp'] / df['min_ambient_temp']
    # df['min_to_max_ratio'] = df['min_ambient_temp'] / df['max_ambient_temp']

    train_df = df.drop(drop_cols, axis=1).copy()
    target_var = df[target_name].values

    feature_cols = train_df.columns

    model.fit(train_df, target_var)

    results = permutation_importance(model, train_df, target_var, scoring=scoring)
    importance = results.importances_mean

    for i, v in zip(feature_cols, importance):
        print('Feature: {}, Score: {:.5f}'.format(i, v))

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([x for x in feature_cols], importance)
        plt.title('Feature importance by permutation importance', fontsize=17)
        plt.xlabel('Feature', fontsize=13)
        for idx, data in zip(feature_cols, importance):
            plt.text(x=idx, y=data, s=f"{round(data, 2)}", fontdict=dict(fontsize=15))
        plt.show()

        if save_plot:
            fig.savefig('feature_importance_plot.png')

    return


if __name__ == '__main__':
    model_names = ['xgboost', 'gradient_boosting', 'random_forest']
    params = []

    for i in model_names:
        with open('{}_params.json'.format(i), 'r') as f:
            model_params = json.load(f)
            params.append(model_params)

    models = [xgb.XGBRegressor(**params[0]),
              GradientBoostingRegressor(**params[1]),
              RandomForestRegressor(**params[2])]

    val_scores = validation(models, model_names)

    print(val_scores)

    print()

    print('Feature importance...')
    print()

    feature_importance_plot(model=models[0], plot=True, save_plot=True)
