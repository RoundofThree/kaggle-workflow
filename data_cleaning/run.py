import pandas as pd 

"""
TODO
"""
def clean_train(filename):
    df = pd.read_csv(filename)
    df.to_csv(f"cleaned_{filename}")

"""
TODO
"""
def clean_test(filename):
    df = pd.read_csv(filename)
    df.to_csv(f"cleaned_{filename}")
