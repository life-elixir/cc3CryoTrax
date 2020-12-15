import pandas as pd
import numpy as np
import json
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb


def regression_precision_error(y_true, y_pred):
    return np.sum(np.absolute(y_true - y_pred) / y_true) / len(y_true)


def validation(params=None, random_state=1):
    default_params = {'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 1000, 'subsample': 0.9}

    drop_cols = ['token',
                 'serial_number',
                 'model',
                 'package_name',
                 'start_time',
                 'exit_time',
                 'min_ambient_temp',
                 '2_8_range_duration']

    target_name = '2_8_range_duration'

    if params:
        default_params = params

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

    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

    model = xgb.XGBRegressor(**default_params)

    # A pipeline to process the feature data
    pipeline = Pipeline(steps=[('knn', imputer),
                               ('m', model)])

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    model_names = ['xgboost']

    scores = []

    for i in model_names:
        scores.append(
            {
                'train_MAE': mean_absolute_error(y_train, train_pred),
                'test_MAE': mean_absolute_error(y_test, test_pred),
                'train_AAD': regression_precision_error(y_train, train_pred),
                'test_AAD': regression_precision_error(y_test, test_pred)
             },
        )
    return pd.DataFrame(scores, index=model_names)


if __name__ == '__main__':
    print(validation())