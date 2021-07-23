import pandas as pd
import os 

#
# Return feature engineered train and test df. 
#
def process_features(train_df: pd.DataFrame, test_df: pd.DataFrame=None):
    # baseline: do nothing 
    # DO SOMETHING
    # remove Name, Sex, Ticket, Cabin, Embarked
    engineered_train_df = train_df.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Sex'], axis=1)
    engineered_train_df['Sex_code'] = train_df['Sex'].replace('male', 0).replace('female', 1).astype('int32')
    if test_df is not None:
        engineered_test_df = test_df.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Sex'], axis=1)
        engineered_test_df['Sex_code'] = test_df['Sex'].replace('male', 0).replace('female', 1).astype('int32')

    # BELOW DEPRECATED
    # save engineered results 
    # print("Debug: ", os.path.dirname(os.path.realpath(__file__)))
    # engineered_root = os.path.dirname(os.path.realpath(__file__))
    # train_df.to_csv(os.path.join(engineered_root, f"engineered_{trainfilename}"))
    # if testfilename:
    #     test_df.to_csv(os.path.join(engineered_root, f"engineered_{testfilename}"))

    return engineered_train_df, engineered_test_df if test_df is not None else None 
