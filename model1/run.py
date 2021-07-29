import pandas as pd
from sklearn.svm import SVC 
import os

#
# Return true if there is a `saved_model` in the module path. 
# 
def is_trained() -> bool:
    return os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_model.joblib"))

#
# Train with train_df and 
# compute the evaluation metrics with cv_df. 
#
def train_with_cv(train_df, cv_df) -> float:
    return 0.0

#
# Save the trained model in saved_model.{ext}
#
def train(train_df):
    # Train the model. 
    # Save the model. 
    saved_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved.model")
    print(f"Debug: Saved model to {saved_model_path}.")


#
# Generate submission.csv in folder of the model module. 
#
def predict(test_df):
    submission_dir = os.path.dirname(os.path.realpath(__file__))
    test_df.to_csv(os.path.join(submission_dir, "submission.csv"))
    print("Debug: Saved predictions to", os.path.join(submission_dir, "submission.csv"))
