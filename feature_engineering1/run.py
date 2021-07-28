import pandas as pd

#
# Return feature engineered train and test df. 
#
def process_features(train_df: pd.DataFrame, test_df: pd.DataFrame=None): 
    # DEPRECATED: 
    # save engineered results 
    # print("Debug: ", os.path.dirname(os.path.realpath(__file__)))
    # engineered_root = os.path.dirname(os.path.realpath(__file__))
    # train_df.to_csv(os.path.join(engineered_root, f"engineered_{trainfilename}"))
    # if testfilename:
    #     test_df.to_csv(os.path.join(engineered_root, f"engineered_{testfilename}"))

    # scaling 
    features_to_scale = ['Fare']
    scale_mean = train_df[features_to_scale].mean(axis=0)
    scale_std = train_df[features_to_scale].std(axis=0)
    train_df[features_to_scale] = (train_df[features_to_scale] - scale_mean) / scale_std 
    if test_df is not None: test_df[features_to_scale] = (test_df[features_to_scale] - scale_mean) / scale_std

    # ratios 

    return train_df, test_df if test_df is not None else None 
