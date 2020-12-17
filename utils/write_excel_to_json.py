import pandas as pd
import json


def convert_excel_to_json(filename, json_file_name, sheet_name=0):
    df = pd.read_excel('../data/' + filename,
                       sheet_name=sheet_name,
                       engine='openpyxl')
    json_str = df.to_json(orient='records')
    unstr_json = json.loads(json_str)
    with open('../data/' + json_file_name, 'w+') as f:
        json.dump(unstr_json, f, indent=1)
    return


if __name__ == '__main__':
    filepath = 'cryotrack_c3_result_4_days_20201014.xlsx'
    json_name = 'c3_json.json'
    sheet = 1
    convert_excel_to_json(filename=filepath,
                          json_file_name=json_name,
                          sheet_name=sheet)