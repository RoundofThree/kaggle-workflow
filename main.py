import argparse
from configparser import ConfigParser
import utils 

#
# Get arguments from command input. 
#
def get_args():
    usage = 'Example: python3 main.py --train train.csv --test test.csv --config ensemble.cfg -v'
    parser = argparse.ArgumentParser(epilog=usage, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", help="train.csv")
    parser.add_argument("--test", help="test.csv") 
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose. Print the score of each model in the ensemble/bagging.")
    parser.add_argument("--config", help="""Config file. Specifies ensemble/bagging, selected feature engineering and models. 
                                        This defaults to ensemble.cfg.""")
    args = parser.parse_args()
    if not args.train:
        parser.error("At least provide a train file.")
    return args 

#
# For all defined models in config file, apply feature engineering and training workflows. 
# If cv flag is set, split into two datasets and perform feature engineering separately. 
# If production flag is set, save the trained model. 
# If both flags are set, split into cv and train, feature engineer both separately, but 
# then train the model from both of them joint and feature engineered as train set. 
#
def train(config: ConfigParser, args):
    mode = config.get('main', "mode")  # "all" or "stacking" or "voting"
    if mode == "stacking":
        train_stacking(config=config, args=args)
    elif mode == "voting":
        train_voting()
    else:
        train_all(config=config, args=args)

#
# Train all the models starting from 0 to n_models-1. 
# If cv flag is set, split a part to cross validate and generate a score.
# If production flag is set, train on all the training dataset, save the models
# to disk, and generate submission.csv. 
#
def train_all(config: ConfigParser, args):
    n_models = config.getint("main", "n_models")
    summary = ""
    for i in range(n_models):
        section = f"model{i}"
        features_module = config.get(section, "features")
        model_module = config.get(section, "model")
        cv = config.getboolean(section, "cv")
        production = config.getboolean(section, "production")
        cv_percent = config.getint(section, "cv_percent")
        cv_times = config.getint(section, "cv_times")
        # check if the model is already trained 
        if utils.is_model_trained(model_module):
            ans = input(f"You already have trained {model_module}, do you want to train from scratch? (yes/no): ")
            if ans.lower() not in ["y", "yes"]: 
                continue # skip this model 
        print("Training...")
        cv_scores = utils.train(features_module, model_module, args.train, cv=cv, production=production, cv_percent=cv_percent, cv_times=cv_times, verbose=args.verbose)
        if args.verbose and cv:
            summary += f"| {model_module} | {cv_scores.mean()} ({cv_scores.std()}) |\n"
    if args.verbose:
        print("\nSummary:\n")
        print(summary)

#
# Use the stacking technique to ensemble all models from 0 to 
# n_models-1. The metalearner is a linear regressor/classifier.
# If cv flag is set, split a part to cross validate and generate a score.
# If production flag is set, train on all the training dataset, save the models
# to disk, and generate submission.csv.
# 
def train_stacking(config: ConfigParser, args):
    n_models = config.getint("main", "n_models")
    cv = config.getboolean("main", "cv")
    production = config.getboolean("main", "production")
    cv_percent = config.getint("main", "cv_percent")
    feature_model_modules = [] 
    for i in range(n_models):
        section = f"model{i}"
        features_module = config.get(section, "features")
        model_module = config.get(section, "model")
        feature_model_modules.append((features_module, model_module))
    cv_score = utils.train_stacking(feature_model_modules, args.train, cv=cv, production=production, cv_percent=cv_percent, verbose=args.verbose)
    if args.verbose and cv:
        print(f"CV score: {cv_score}")

#
# Use the voting technique to ensemble all models from 0 to 
# n_models-1. The metalearner is a linear regressor/classifier.
# If cv flag is set, split a part to cross validate and generate a score.
# If production flag is set, train on all the training dataset, save the models
# to disk, and generate submission.csv.
#
def train_voting():
    pass 

#
# Generate submission.csv for each corresponding model in their folders. 
# 
def predict(config: ConfigParser, args):
    mode = config.get("main", "mode")
    if mode == "stacking":
        predict_stacking(config, args)
    elif mode == "voting":
        predict_voting(config, args)
    else:
        predict_all(config, args)
    

#
# Generate submission.csv for mode=all.
#
def predict_all(config: ConfigParser, args):
    n_models = config.getint("main", "n_models")
    for i in range(n_models):
        section = f"model{i}"
        features_module = config.get(section, "features")
        model_module = config.get(section, "model")
        production = config.getboolean(section, "production")
        if not production: continue 
        # generate submission.csv in model folder 
        print("Predicting...")
        utils.predict(features_module, model_module, args.train, args.test, verbose=args.verbose)

# 
# Generate submission.csv for stacking model. 
#
def predict_stacking(config: ConfigParser, args):
    n_models = config.getint("main", "n_models")
    if not config.getboolean("main", "production"):
        return 
    feature_model_modules = [] 
    for i in range(n_models):
        section = f"model{i}"
        features_module = config.get(section, "features")
        model_module = config.get(section, "model")
        feature_model_modules.append((features_module, model_module))
    utils.predict_stacking(feature_model_modules, args.train, args.test, save=True)

#
# Generate submission.csv for voting model.
#
def predict_voting(config: ConfigParser, args):
    n_models = config.getint("main", "n_models")
    if not config.getboolean("main", "production"):
        return 


if __name__ == '__main__':
    args = get_args()
    config = ConfigParser()
    config.read(args.config or "ensemble.cfg")
    if args.train:
        train(config, args)
    if args.test:
        predict(config, args)
    print("*************")
    print("Best of luck!")
    print("*************")
