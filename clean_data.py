import data_cleaning.run
import argparse

def get_args():
    usage = "Example:  python3 clean_data.py --train train.csv --test test.csv"
    parser = argparse.ArgumentParser(epilog=usage, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", help="train.csv")
    parser.add_argument("--test", help="test.csv")
    # data cleaning is different from train and test, eg. test cannot remove rows 
    args = parser.parse_args()
    if not args.train and not args.test:
        print("Please provide at least a train.csv or test.csv.")
    return args 

if __name__ == '__main__':
    args = get_args()
    # call clean(train) and clean(test) if not None
    # save the cleaned data in cleaned_train.csv 
    if args.train: data_cleaning.run.clean_train(args.train)
    if args.test: data_cleaning.run.clean_test(args.test)
