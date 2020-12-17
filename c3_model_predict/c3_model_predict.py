import warnings
import pickle
import pandas as pd
import numpy as np
import json


class InputValueWarning(UserWarning):
    pass


class RequiredFeatureMissingWarning(UserWarning):
    pass


def input_reformat(data: list):
    """
    Reformat the column names from 'A B' to 'a_b', i.e. lower case and connect with _.
    param data: list
    return: list
    """
    for pred in data:
        for key in list(pred.keys()):
            name = '_'.join(key.lower().split())
            pred[name] = pred.pop(key)
    return data


def check_c3_input_data(lst, input_features, required_features):
    """
    Check raw held out data, perform necessary processing before prediction
    :param lst: (list) input data
    :param input_features: (list) features needed for the model to make predictions
    :param required_features: (list) required features, lack of these features will weaken the
    model predictive power and therefore will give warming messages
    :return: (pd.DataFrame)
    """
    lst = input_reformat(lst)

    if lst:
        for data in lst:
            missing_required_feature_count = 0
            for feature in input_features:
                value = data.get(feature)
                if not isinstance(value, (float, int)):
                    if feature in required_features:
                        missing_required_feature_count += 1
                    data[feature] = np.nan
                    message = f'{feature} is not of int or float, it is of {type(value)}. The prediction may not be ' \
                              f'as accurate. '
                    warnings.warn(message, InputValueWarning)

            if missing_required_feature_count > 0:
                message = f'The input data miss {missing_required_feature_count} required features, or the required ' \
                          f'features are of the wrong data type, investigation is ' f'required. '
                warnings.warn(message, RequiredFeatureMissingWarning)

        return pd.DataFrame(lst, dtype=float)

    else:
        raise ValueError('There is no data inflow ')


def load_model():
    model = pickle.load(open('../c3_model_train/c3_model.pickle', 'rb'))
    print('Model loaded')
    model_columns = pickle.load(open('../c3_model_train/c3_model_columns.pickle', 'rb'))
    print('Model columns loaded')
    return model, model_columns


def c3_model_predict(json_):
    model, model_cols = load_model()

    if model:
        required_features = ['max_ambient_temp',
                             'ambient_mkt_value']

        model_cols.remove('max_to_min_ambient_ratio')

        query = check_c3_input_data(json_, model_cols, required_features)

        model_cols.append('min_ambient_temp')

        query = query.reindex(columns=model_cols, fill_value=0)

        query['max_to_min_ambient_ratio'] = query['max_ambient_temp'] / query['min_ambient_temp']

        query.drop('min_ambient_temp', axis=1, inplace=True)

        print('Predicting')

        prediction = model.predict(query)

        return prediction

    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':
    input_sample = json.load(open('../data/c3_json.json', 'r'))[:10]
    print(json.dumps(input_sample, indent=2))
    print(c3_model_predict(input_sample))