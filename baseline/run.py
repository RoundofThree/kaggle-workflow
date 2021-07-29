import pandas as pd
import xgboost as xgb 
from sklearn.metrics import accuracy_score
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
def train_with_cv(train_df: pd.DataFrame, cv_df: pd.DataFrame) -> float:
    train_X, train_y = train_df.drop('Survived', axis=1), train_df['Survived']
    cv_X, cv_y = cv_df.drop('Survived', axis=1), cv_df['Survived']
    ### XGBoost basic with early stopping 
    model = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.01, use_label_encoder=False, eval_metric="error")
    model.fit(train_X, train_y, early_stopping_rounds=50, eval_set=[(train_X, train_y), (cv_X, cv_y)], verbose=False)
    # print(f"Best ntree_limit: {model.best_ntree_limit}") # 104
    ### metrics 
    predictions = model.predict(cv_X)
    score = accuracy_score(cv_y, predictions)
    ### TODO: if verbose, print the features importance
    return score

#
# Save the trained model in saved_model.{ext}
# 
def train(train_df: pd.DataFrame, save=True):
    train_X, train_y = train_df.drop('Survived', axis=1), train_df['Survived']
    # Train the model. 
    model = xgb.XGBClassifier(n_estimators=104, max_depth=6, learning_rate=0.01, use_label_encoder=False, eval_metric="error")
    model.fit(train_X, train_y)
    # print(f"Best ntree limit: {model.best_ntree_limit}")
    # Save the model
    if save:
        saved_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_model.json")
        model.save_model(saved_model_path)
        print(f"[Debug]: Saved model to {saved_model_path}.")

#
# Generate submission.csv in folder of the model module. 
# If save=False, return array of predictions. 
#
def predict(test_df, save=True):
    # load model 
    model = xgb.XGBClassifier()
    saved_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_model.json")
    model.load_model(saved_model_path)
    # generate predictions
    predictions = model.predict(test_df)
    if save:
        submission = pd.DataFrame({'Survived': predictions}, index=test_df.index)
        submission_dir = os.path.dirname(os.path.realpath(__file__))
        submission.to_csv(os.path.join(submission_dir, "submission.csv"))
        print("[Debug]: Saved predictions to", os.path.join(submission_dir, "submission.csv"))
    else:
        return predictions 
