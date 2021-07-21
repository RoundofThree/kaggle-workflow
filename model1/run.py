import pandas as pd 
import os

"""
Return true if there is a `saved_model.h5` in the module path. 
"""
def is_trained() -> bool:
    return False 

def train_with_cv(trainfilename) -> float:
    return 0.0

"""
Save the trained model in saved_model.{ext}
"""
def train(trainfilename):
    # Train the model. 
    # Save the model. 
    saved_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_model")
    print(f"Debug: Saved model to {saved_model_path}.")


"""
Generate submission.csv in folder of the model module. 
"""
def predict(testfilename):
    test_df = pd.read_csv(testfilename)
    submission_dir = os.path.dirname(os.path.realpath(__file__))
    test_df.to_csv(os.path.join(submission_dir, "submission.csv"))
    print("Debug: Saved predictions to", os.path.join(submission_dir, "submission.csv"))