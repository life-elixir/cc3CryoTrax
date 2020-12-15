import pandas as pd
import numpy as np
import json
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb


def regression_precision_error(y_true, y_pred):
    return np.sum(np.absolute(y_true - y_pred) / y_true) / len(y_true)


def validation(models, model_names, random_state=1):

    drop_cols = ['token',
                 'serial_number',
                 'model',
                 'package_name',
                 'start_time',
                 'exit_time',
                 'min_ambient_temp',
                 '2_8_range_duration']

    target_name = '2_8_range_duration'

    # Load json
    df = pd.read_json('../data/c3_json.json', orient='records')

    # Engineer features
    df['max_to_min_ratio'] = df['max_ambient_temp'] / df['min_ambient_temp']
    df['min_to_max_ratio'] = df['min_ambient_temp'] / df['max_ambient_temp']

    train_df = df.drop(drop_cols, axis=1).copy()
    target_var = df[target_name].values

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
