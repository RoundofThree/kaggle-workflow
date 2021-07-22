import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from importlib import import_module

"""
Return true if the model in `model_module` has a saved model file. 
"""
def is_model_trained(model_module):
    is_trained = get_function(model_module, "is_trained")
    return is_trained()

"""
Generator.
Generate pd.DataFrame of feature engineered train data and cv data (optional) and test data. 
"""
def feature_engineer(features_module, trainfilename, testfilename=None, cv=True, cv_percent=20, cv_times=5) -> tuple:
    process_features = get_function(features_module, "process_features")
    train_df = pd.read_csv(trainfilename)
    test_df = pd.read_csv(testfilename) if testfilename else None 
    if not cv:
        engineered_train_df, engineered_test_df = process_features(train_df, test_df)
        yield engineered_train_df, None, engineered_test_df 
    else:
        if 100%cv_percent == 0 and cv_times == 100//cv_percent:
            # kfold
            # split train_df into cv_times parts
            batch_size = len(train_df) // cv_times
            for i in range(cv_times):
                splitted_train_df = pd.concat([train_df.iloc[0:i*batch_size], train_df.iloc[(i+1)*batch_size:]])
                splitted_cv_df = train_df.iloc[i*batch_size:(i+1)*batch_size]
                engineered_train_df, engineered_cv_df = process_features(splitted_train_df, splitted_cv_df)
                yield engineered_train_df, engineered_cv_df, None 
        else:
            # choose randomly for cv_times
            for i in range(cv_times):
                splitted_train_df, splitted_cv_df = train_test_split(train_df, test_size=cv_percent/100)
                engineered_train_df, engineered_cv_df = process_features(splitted_train_df, splitted_cv_df)
                yield engineered_train_df, engineered_cv_df, None


"""
Deprecated. Now engineered data is passed in memory to avoid complications. 
Return true if there is engineered_{filename}.csv in the features_module folder.
"""
# def is_engineered(features_module, filename) -> bool:
#     package = features_module.split(".")[0]
#     if os.path.isfile(os.path.abspath(f"{package}/engineered_{filename}")):
#         # print(f"Debug: Found {os.path.abspath(f'{package}/engineered_{filename}')}")
#         return True 
#     else: 
#         return False 

"""
If cv=True, return cross validation error else return None. 
If production=True, generate saved_model.{ext}.
"""
def train(features_module, model_module, trainfilename, cv=True, production=False, cv_percent=20, cv_times=5):
    cv_score = None 
    if cv:
        cv_score = 0 
        train_with_cv = get_function(model_module, "train_with_cv")
        generator = feature_engineer(features_module, trainfilename, cv=True, cv_percent=cv_percent, cv_times=cv_times)
        for _ in range(cv_times):
            train_df, cv_df, _ = next(generator)
            curr_cv_score = train_with_cv(train_df, cv_df)
            cv_score += curr_cv_score
        cv_score /= cv_times
    # if production, save the trained model on the full train data 
    if production:
        generator = feature_engineer(features_module, trainfilename, cv=False)
        full_train_df, _, _ = next(generator)
        train_model = get_function(model_module, "train")
        train_model(full_train_df)
    return cv_score 

    
"""
Generate submission.csv in folder model_module. 
"""
def predict(features_module, model_module, trainfilename, testfilename):
    predict_test = get_function(model_module, "predict")
    generator = feature_engineer(features_module, trainfilename, testfilename=testfilename, cv=False)
    _, _, test_df = next(generator)
    predict_test(test_df)


##### Reflection functions

def get_function(module_path, fn_name):
    try:
        module = import_module(module_path)
    except ModuleNotFoundError:
        raise ValueError(f"Cannot find module {module_path}")
    try: 
        return getattr(module, fn_name)
    except AttributeError:
        raise ValueError(f"Cannot find function {fn_name} in module {module_path}")




