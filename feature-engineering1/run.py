import pandas as pd
import os 
"""
Generate engineered_{trainfilename} and engineered_{testfilename}. 
"""
def process_features(trainfilename, testfilename=None):
    # do something 
    # baseline: do nothing 
    train_df = pd.read_csv(trainfilename)
    if testfilename:
        test_df = pd.read_csv(testfilename)
    # save engineered results 
    # print("Debug: ", os.path.dirname(os.path.realpath(__file__)))
    engineered_root = os.path.dirname(os.path.realpath(__file__))
    train_df.to_csv(os.path.join(engineered_root, f"engineered_{trainfilename}"))
    if testfilename:
        test_df.to_csv(os.path.join(engineered_root, f"engineered_{testfilename}"))
