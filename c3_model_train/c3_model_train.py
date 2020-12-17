import pickle
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb


def c3_model_train():
    params = {"learning_rate": 0.01,
              "n_estimators": 100,
              "subsample": 1.0}

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
    df['max_to_min_ambient_ratio'] = df['max_ambient_temp'] / df['min_ambient_temp']
    
    df.rename(columns={'ambient_MKT_value': 'ambient_mkt_value'}, inplace=True)

    train_df = df.drop(drop_cols, axis=1)
    target_var = df[target_name].values

    model = xgb.XGBRegressor(**params)

    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

    pipeline = Pipeline(steps=[('knn', imputer),
                               ('m', model)])

    pipeline.fit(train_df, target_var)

    with open('c3_model.pickle', 'wb') as f:
        pickle.dump(pipeline, f)

    with open('c3_model_columns.pickle', 'wb') as f:
        pickle.dump(train_df.columns.to_list(), f)

    msg = 'Successfully re-trained model and generated new .pkl files for c3_model'
    print(msg)

    return msg


if __name__ == '__main__':
    c3_model_train()