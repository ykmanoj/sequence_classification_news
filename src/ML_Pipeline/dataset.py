import glob
from json import loads

import pandas as pd
from ML_Pipeline.Constants import *



def read_data(filename,**kwargs):
    raw_data=pd.read_csv(input_dir+filename,**kwargs)
    return raw_data

def read_directory(dirname,**kwargs):
    l = [pd.read_csv(filename,**kwargs) for filename in glob.glob(dirname+"*")]
    df = pd.concat(l, axis=0)
    return df

def collect_all_datasets(filename):
    if filename=='train':
        train_data = read_data(train_data_filename, names=column_names)
        valid_data = read_data(valid_data_filename, names=column_names)
        train_df = pd.concat([train_data, valid_data])
        return train_df
    if filename == 'test':
        test_data = read_data(test_data_filename, names=column_names)
        return test_data
    else:
        train_data = read_data(train_data_filename, names=column_names)
        valid_data = read_data(valid_data_filename, names=column_names)
        train_df = pd.concat([train_data, valid_data])
        test_data = read_data(test_data_filename, names=column_names)
        return train_df,test_data

def read_json_request(json_str):
    js_obj=loads(json_str)
    df = pd.DataFrame.from_dict(js_obj)
    return df


