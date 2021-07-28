import pandas as pd 
import numpy as np

"""
TODO: See analysis.ipynb
"""
def clean(trainfilename, testfilename=None):
    train_df = pd.read_csv(trainfilename, index_col="PassengerId")
    test_df = pd.read_csv(testfilename, index_col="PassengerId") 
    # missing values 
    # Age: impute mean 
    age_mean = train_df["Age"].mean()
    train_df["Age"].fillna(age_mean, inplace=True) 
    test_df["Age"].fillna(age_mean, inplace=True)
    # Fare: by group Pclass
    fare_by_pclass = train_df.groupby("Pclass")["Fare"].mean()
    test_df["Fare"].fillna(test_df.apply(lambda r: fare_by_pclass[r['Pclass']], axis=1), inplace=True)
    # Embarked 
    train_df["Embarked"].fillna("S", inplace=True)
    # typos 
    train_df.drop(labels=train_df.loc[train_df['Fare'] == 0].index, axis=0, inplace=True) # Fare == 0
    # encoding 
    bins = [0,17,26,34,50,63,100]
    train_df['AgeGroup'] = pd.cut(train_df['Age'], bins, labels=[0,1,2,3,4,5])
    test_df['AgeGroup'] = pd.cut(test_df['Age'], bins, labels=[0,1,2,3,4,5])
    train_df['HasCabin'] = train_df['Cabin'].notna()
    test_df['HasCabin'] = test_df['Cabin'].notna()
    train_df['Sex'] = (train_df['Sex'] == 'male').astype(np.int32)
    test_df['Sex'] = (test_df['Sex'] == 'female').astype(np.int32)
    train_embarked_oh = pd.get_dummies(train_df['Embarked'], prefix="Embarked")
    test_embarked_oh = pd.get_dummies(test_df['Embarked'], prefix='Embarked')
    train_df = pd.merge(left=train_df, right=train_embarked_oh, left_index=True, right_index=True)
    test_df = pd.merge(left=test_df, right=test_embarked_oh, left_index=True, right_index=True)
    # removing 
    train_df.drop(["Name", "Ticket", "Cabin", "Age", "Embarked"], axis=1, inplace=True)
    test_df.drop(["Name", "Ticket", "Cabin", "Age", "Embarked"], axis=1, inplace=True)
    # should I remove too high fares? Not for now but maybe 
    # scaling should be done in feature engineering module 
    # features_to_scale = ['Fare']
    # scale_mean = train_df[features_to_scale].mean(axis=0)
    # scale_std = train_df[features_to_scale].std(axis=0)
    # train_df[features_to_scale] = (train_df[features_to_scale] - scale_mean) / scale_std 
    # test_df[features_to_scale] = (test_df[features_to_scale] - scale_mean) / scale_std
    # to csv
    train_df.to_csv(f"cleaned_{trainfilename}", index="PassengerId")
    if testfilename: test_df.to_csv(f"cleaned_{testfilename}", index="PassengerId")

