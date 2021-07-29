import pandas as pd

#
# Return feature engineered train and test df. 
#
def process_features(train_df: pd.DataFrame, test_df: pd.DataFrame=None): 
    # scaling 
    features_to_scale = ['Fare']
    scale_mean = train_df[features_to_scale].mean(axis=0)
    scale_std = train_df[features_to_scale].std(axis=0)
    train_df[features_to_scale] = (train_df[features_to_scale] - scale_mean) / scale_std 
    if test_df is not None: test_df[features_to_scale] = (test_df[features_to_scale] - scale_mean) / scale_std

    # ratios 

    return train_df, test_df if test_df is not None else None 
