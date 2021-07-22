import pandas as pd
import os 
"""
Generate engineered_{trainfilename} and engineered_{testfilename}. 
"""
def process_features(train_df, test_df=None):
    # baseline: do nothing 
    # DO SOMETHING

    # BELOW DEPRECATED
    # save engineered results 
    # print("Debug: ", os.path.dirname(os.path.realpath(__file__)))
    # engineered_root = os.path.dirname(os.path.realpath(__file__))
    # train_df.to_csv(os.path.join(engineered_root, f"engineered_{trainfilename}"))
    # if testfilename:
    #     test_df.to_csv(os.path.join(engineered_root, f"engineered_{testfilename}"))

    return train_df, test_df 
