import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

#
# Return true if there is a `saved_model.json` in the module path. 
#
def is_trained() -> bool:
    return os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_model.json"))

#
# Train with train_df and 
# compute the evaluation metrics with cv_df. 
#
def train_with_cv(train_df, cv_df) -> float:
    train_X, train_y = train_df.drop('Survived', axis=1), train_df['Survived']
    cv_X, cv_y = cv_df.drop('Survived', axis=1), cv_df['Survived']
    ### XGBoost basic with early stopping 
    params = {'n_estimators': 550, 'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.01, 'verbosity': 0, 'n_jobs': 4}
    bst = XGBClassifier(**params, use_label_encoder=False)
    bst.fit(train_X, train_y, eval_metric='logloss', eval_set=[(cv_X, cv_y)], early_stopping_rounds=5, verbose=False)
    # print the best n_estimators: 
    # print(bst.best_ntree_limit)
    predictions = bst.predict(cv_X) # the output is a numpy array 
    assert predictions.shape == cv_y.shape
    ### metrics 
    score = accuracy_score(cv_y, predictions)
    return score

#
# Save the trained model in saved_model.{ext}
# 
def train(train_df):
    train_X, train_y = train_df.drop('Survived', axis=1), train_df['Survived']
    # Train the model. 
    params = {'n_estimators': 300, 'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.01, 'verbosity': 0, 'n_jobs': 4}
    bst = XGBClassifier(**params, use_label_encoder=False)
    # bst._le = LabelEncoder().fit([1, 0])
    bst.fit(train_X, train_y, verbose=False) # stop at when last cv validation stopped 
    # Save the model
    saved_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_model.json")
    bst.save_model(saved_model_path)
    print(f"Debug: Saved model to {saved_model_path}.")


#
# Generate submission.csv in folder of the model module. 
#
def predict(test_df):
    # load model 
    bst = XGBClassifier()
    saved_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_model.json")
    bst.load_model(saved_model_path)
    # print("Debug:", bst)
    # generate predictions
    predictions = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': bst.predict(test_df)})
    submission_dir = os.path.dirname(os.path.realpath(__file__))
    predictions.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)
    print("Debug: Saved predictions to", os.path.join(submission_dir, "submission.csv"))
