import argparse
from configparser import ConfigParser
import utils 

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

"""
For all defined models in config file, apply feature engineering and training workflows. 
If cv flag is set, split into two datasets and perform feature engineering separately. 
If production flag is set, save the trained model. 
If both flags are set, split into cv and train, feature engineer both separately, but 
then train the model from both of them joint and feature engineered as train set. 
"""
def train(config: ConfigParser, args):
    mode = config.get('main', "mode")  # "ensemble" or "bagging". TODO: for now it doesn't make any difference 
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
        cv_score = utils.train(features_module, model_module, args.train, cv=cv, production=production, cv_percent=cv_percent, cv_times=cv_times, verbose=args.verbose)
        if args.verbose and cv:
            summary += f"| {model_module} | {cv_score} |\n"
    print("\nSummary:\n")
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
        # if not utils.is_engineered(features_module, args.test):
        #     print("Please run main.py --train train.csv --test test.csv first.")
        #     continue 

        # generate submission.csv in model folder 
        print("Predicting...")
        utils.predict(features_module, model_module, args.train, args.test, verbose=args.verbose)


if __name__ == '__main__':
    args = get_args()
    config = ConfigParser()
    config.read(args.config or "ensemble.cfg")
    if args.train:
        train(config, args)
    if args.test:
        predict(config, args)
    print("Best of luck!")
