import pandas as pd
import numpy as np
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split
from importlib import import_module
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

#
# Return true if the model in `model_module` has a saved model file. 
#
def is_model_trained(model_module):
    is_trained = get_function(model_module, "is_trained")
    return is_trained()

#
# Generator.
# Generate pd.DataFrame of feature engineered train data and cv data (optional) and test data. 
#
def feature_engineer(features_module, trainfilename, testfilename=None, cv=True, cv_percent=20, cv_times=5) -> tuple:
    process_features = get_function(features_module, "process_features")
    train_df = pd.read_csv(trainfilename, index_col="PassengerId")
    test_df = pd.read_csv(testfilename, index_col="PassengerId") if testfilename else None 
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

#
# If cv=True, return cross validation error else return None. 
# If production=True, generate saved_model.{ext}.
#
def train(features_module, model_module, trainfilename, cv=True, production=False, cv_percent=20, cv_times=5, verbose=False):
    cv_scores = None 
    if cv:
        cv_scores = []
        train_with_cv = get_function(model_module, "train_with_cv")
        generator = feature_engineer(features_module, trainfilename, cv=True, cv_percent=cv_percent, cv_times=cv_times)
        for i in range(cv_times):
            train_df, cv_df, _ = next(generator)
            curr_cv_score = train_with_cv(train_df, cv_df)
            if verbose: print(f"[Validation {i}]: {curr_cv_score}")
            cv_scores.append(curr_cv_score)
        print("[Debug]: ", cv_scores)
        cv_scores = np.asarray(cv_scores)
    # if production, save the trained model on the full train data 
    if production:
        generator = feature_engineer(features_module, trainfilename, cv=False)
        full_train_df, _, _ = next(generator)
        train_model = get_function(model_module, "train")
        train_model(full_train_df)
    return cv_scores

#
# If cv flag is set, return cv_score else return None. 
# If production flag is set, 
#
def train_stacking(feature_model_modules, trainfilename, cv=True, production=False, cv_percent=20, verbose=False):
    train_df = pd.read_csv(trainfilename, index_col="PassengerId")
    # shuffle train_df but preserve index 
    train_df = train_df.sample(frac=1)
    n_models = len(feature_model_modules)
    if cv: 
        layer_2_train_df = pd.DataFrame({})
        layer_2_cv_df = pd.DataFrame({})
        train_df, cv_df = train_test_split(train_df, test_size=cv_percent/100)
        batch_size = train_df.shape[0] // n_models
        for i, (features_module, model_module) in enumerate(feature_model_modules):
            process_features = get_function(features_module, "process_features")
            engineered_train_df, engineered_cv_df = process_features(train_df, cv_df)
            batch = engineered_train_df.iloc[i*batch_size:(i+1)*batch_size+1] if i != n_models-1 else engineered_train_df.iloc[i*batch_size:]
            train_model = get_function(model_module, "train")
            # fit
            train_model(engineered_train_df.drop(batch.index)) # will save the model 
            # get layer_2_train
            predict_test = get_function(model_module, "predict")
            layer_2_train_df[model_module] = predict_test(engineered_train_df.drop(['Survived'], axis=1), save=False)
            layer_2_cv_df[model_module] = predict_test(engineered_cv_df.drop(['Survived'], axis=1), save=False)
        metalearner = LogisticRegression()
        metalearner.fit(layer_2_train_df, train_df['Survived'])
        predictions = metalearner.predict(layer_2_cv_df)
        score = accuracy_score(cv_df['Survived'], predictions)
        return score 
    if production:
        train_df = pd.read_csv(trainfilename, index_col="PassengerId")
        layer_2_train_df = pd.DataFrame({})
        batch_size = train_df.shape[0] // n_models
        # training loop 
        for i, (features_module, model_module) in enumerate(feature_model_modules):
            # if model has already been trained, ask to train again 
            if is_model_trained(model_module):
                process_features = get_function(features_module, "process_features")
                engineered_train_df, _ = process_features(train_df)
                ans = input(f"Already trained {model_module}, do you want to train again? (yes/no): ")
                if ans.lower() in ["y", "yes"]:
                    batch = engineered_train_df.iloc[i*batch_size:(i+1)*batch_size+1] if i != n_models-1 else engineered_train_df.iloc[i*batch_size:]
                    train_model = get_function(model_module, "train")
                    # fit
                    train_model(engineered_train_df.drop(batch.index)) # will save the model 
            # get layer_2_train
            predict_test = get_function(model_module, "predict")
            layer_2_train_df[model_module] = predict_test(engineered_train_df.drop(['Survived'], axis=1), save=False)

        metalearner = LogisticRegression(solver='lbfgs')
        metalearner.fit(layer_2_train_df, train_df['Survived'])
        # save metalearner model 
        dump(metalearner, "stacking_metalearner.joblib")
    
    
#
# Generate submission.csv in folder model_module. 
#
def predict(features_module, model_module, trainfilename, testfilename, verbose=False, save=True):
    predict_test = get_function(model_module, "predict")
    generator = feature_engineer(features_module, trainfilename, testfilename=testfilename, cv=False)
    _, _, test_df = next(generator)
    predict_test(test_df, save=save)

#
# Generate stacking_submission.csv in root folder.
#
def predict_stacking(feature_model_modules, trainfilename, testfilename, verbose=False, save=True):
    layer_2_test_df = pd.DataFrame({})
    original_test_df = pd.read_csv(testfilename, index_col="PassengerId")
    for features_module, model_module in feature_model_modules:
        predict_test = get_function(model_module, "predict")
        generator = feature_engineer(features_module, trainfilename, testfilename=testfilename, cv=False)
        _, _, test_df = next(generator)
        layer_2_test_df[model_module] = predict_test(test_df, save=False) 
    metalearner = LogisticRegression()
    metalearner = load("stacking_metalearner.joblib")
    layer_2_test_df.index = original_test_df.index
    predictions = metalearner.predict(layer_2_test_df)
    submission = pd.DataFrame({'Survived': predictions}, index=original_test_df.index)
    if save:
        submission.to_csv("stacking_submission.csv")

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




