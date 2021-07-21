import argparse
from configparser import ConfigParser
import utils 

def get_args():
    usage = 'Example: TODO'
    parser = argparse.ArgumentParser(epilog=usage, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", help="train.csv")
    parser.add_argument("--test", help="test.csv") 
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose. Print the score of each model in the ensemble/bagging.")
    parser.add_argument("--config", help="""Config file. Specifies ensemble/bagging, selected feature engineering and models. 
                                        This defaults to ensemble.cfg.""")
    args = parser.parse_args()
    if not (args.train or args.test):
        parser.error("Either provide a train file or a test file.")
    return args 

"""
You already have trained models, do you want to train again? (yes/no): y
Training...
(If verbose, should show the cross validation scores for each model)
Saved the trained models. 
"""
def train(config: ConfigParser, args):
    mode = config.get('main', "mode")  # "ensemble" or "bagging". TODO: for now it doesn't make difference 
    n_models = config.getint("main", "n_models")
    summary = ""
    
    for i in range(n_models):
        section = f"model{i}"
        features_module = config.get(section, "features")
        model_module = config.get(section, "model")
        cv = config.getboolean(section, "cv")
        production = config.getboolean(section, "production")
        # check if the model is already trained 
        if utils.is_model_trained(model_module):
            ans = input(f"You already have trained {model_module}, do you want to train from scratch? (yes/no): ")
            if ans.lower() not in ["y", "yes"]: 
                continue # skip this model 
        
        # generate engineered_train.csv and engineered_test.csv 
        utils.feature_engineer(features_module, args.train, args.test)
        # model.train given engineered_train.csv 
        print("Training...")
        cv_score = utils.train(model_module, args.train, cv=cv, production=production) # first with cv set, then train without cv set 
        if args.verbose and cv:
            print(f"| {model_module} | {cv_score} |")
            summary += f"\n| {model_module} | {cv_score} |"
    print("\nSummary:")
    print(summary)


"""
Generate submission.csv for each corresponding model in their folders. 
"""
def predict(config: ConfigParser, args):
    n_models = config.getint("main", "n_models")

    for i in range(n_models):
        section = f"model{i}"
        features_module = config.get(section, "features")
        model_module = config.get(section, "model")
        production = config.getboolean(section, "production")
        if not production: continue 
        # check if there are engineered_test.csv 
        if not utils.is_engineered(features_module, args.test):
            print("Please run main.py --train train.csv --test test.csv first.")
            continue 
        # generate submission.csv in model folder 
        print("Predicting...")
        utils.predict(model_module, args.test)


if __name__ == '__main__':
    args = get_args()
    config = ConfigParser()
    config.read(args.config or "ensemble.cfg")
    if args.train:
        train(config, args)
    if args.test:
        predict(config, args)
    print("Best of luck!")
