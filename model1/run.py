from operator import sub
import pandas as pd
from sklearn.metrics.classification import accuracy_score
from sklearn.svm import SVC 
import os
from joblib import load, dump 

#
# Return true if there is a `saved_model` in the module path. 
# 
def is_trained() -> bool:
    return os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_model.joblib"))


#
# Train with train_df and 
# compute the evaluation metrics with cv_df. 
#
def train_with_cv(train_df: pd.DataFrame, cv_df: pd.DataFrame) -> float:
    svc = SVC(C=1, probability=True, gamma='auto')
    train_X, train_y = train_df.drop(['Survived'], axis=1), train_df['Survived']
    cv_X, cv_y = cv_df.drop(['Survived'], axis=1), cv_df['Survived']
    # fit 
    svc.fit(train_X, train_y)
    # predict 
    predictions = svc.predict(cv_X)
    score = accuracy_score(cv_y, predictions)
    return score 

#
# Save the trained model in saved_model.{ext}
#
def train(train_df: pd.DataFrame, save=True):
    # Train the model. 
    svc = SVC(C=1, probability=True, gamma='auto')
    svc.fit(train_df.drop(['Survived'], axis=1), train_df['Survived'])
    # Save the model. 
    if save:
        saved_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_model.joblib")
        dump(svc, saved_model_path)
        print(f"[Debug]: Saved model to {saved_model_path}.")


#
# Generate submission.csv in folder of the model module. 
#
def predict(test_df, save=True):
    # load model 
    saved_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_model.joblib")
    svc = load(saved_model_path)
    submission = svc.predict(test_df)
    if save:
        submission = pd.DataFrame({'Survived': submission}, index=test_df.index)
        submission_dir = os.path.dirname(os.path.realpath(__file__))
        submission.to_csv(os.path.join(submission_dir, "submission.csv"))
        print("[Debug]: Saved predictions to", os.path.join(submission_dir, "submission.csv"))
    else:
        return submission
