import os 
from importlib import import_module

"""
Return true if the model in `model_module` has a saved model file. 
"""
def is_model_trained(model_module):
    # get function model.is_trained
    is_trained = get_function(model_module, "is_trained")
    return is_trained() 

"""
Generate engineered_{trainfilename}.csv and engineered_{testfilename}.csv 
in folder {features_module}/. 
"""
def feature_engineer(features_module, trainfilename, testfilename=None) -> None:
    process_features = get_function(features_module, "process_features")
    process_features(trainfilename, testfilename) 

"""
Return true if there is engineered_{filename}.csv in the features_module folder.
"""
def is_engineered(features_module, filename) -> bool:
    package = features_module.split(".")[0]
    if os.path.isfile(os.path.abspath(f"{package}/engineered_{filename}")):
        # print(f"Debug: {os.path.abspath(f'{package}/engineered_{filename}')}")
        return True 
    else: 
        return False 

"""
If cv=True, return cross validation error. 
If production=True, generate saved_model.{ext}.
"""
def train(model_module, trainfilename, cv=True, production=False):
    cv_score = None 
    if cv: 
        train_with_cv = get_function(model_module, "train_with_cv")
        cv_score = train_with_cv(trainfilename)
    if production:
        train_model = get_function(model_module, "train")
        train_model(trainfilename) # generate saved_model  
    return cv_score 
    
"""
Generate submission.csv in folder model_module. 
"""
def predict(model_module, testfilename):
    predict = get_function(model_module, "predict")
    predict(testfilename)


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




