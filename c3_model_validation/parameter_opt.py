import numpy as np
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost

params_xgboost = dict(
    learning_rate=np.linspace(0.01, 1., 10),
    max_depth=list(range(1, 3)),
    n_estimators=[100, 500, 1000, 1500],
    subsample=[.7, .8, .9]
)

params_random_forest = dict(
    n_estimators=[100, 500, 1000, 1500],
    criterion=['mse', 'mae'],
    max_depth=list(range(1, 3)),
)

params_gradient_boosting = dict(
    learning_rate=np.linspace(0.01, 1., 10),
    n_estimators=[100, 500, 1000, 1500],
    subsample=[.7, .8, .9, 1.],
)

params = [params_xgboost, params_gradient_boosting, params_random_forest]

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

xgb = xgboost.XGBRegressor()
r = RandomForestRegressor()
g = GradientBoostingRegressor()

models = [xgb, g, r]
model_names = ['xgboost', 'gradient_boosting', 'random_forest']

for i in range(len(models)):
    print('Hyperparameter tuning: {}...'.format(model_names[i]))
    clf = GridSearchCV(models[i],
                       param_grid=params[i],
                       scoring='neg_mean_absolute_error',
                       n_jobs=-1,
                       return_train_score=True,
                       verbose=1)

    clf.fit(train_df, target_var)
    print()
    print('The best scores for {} is {}.'.format(model_names[i], clf.best_score_))
    print()

    with open('{}_params.json'.format(model_names[i]), 'w+') as f:
        json.dump(clf.best_params_, f)



