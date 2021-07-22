# Kaggle worflow 

This is a template folder I use to generate submissions in Kaggle. 

## Input 

Data cleaning and preprocessing can be done in a separate step, then save the cleaned data in a separate csv file. 

```
Usage:
    python3 clean_data.py --train train.csv --test test.csv 

Outputs clean_train.csv and clean_test.csv. 
```

Usually, I can split the training and predicting process, so I splitted the command into two modes: train and test. 
Caveat: some feature engineering techniques such as grouping and target encoding make test feature engineering depend on 
train feature engineering values. Therefore, train mode should save engineered_train.csv and engineered_test.csv in the
corresponding folder. 

Provide a train.csv to train. 

```
Usage: 
    python3 main.py --train train.csv --test test.csv

You already have trained model1, do you want to train again? (yes/no): y
You already have trained model2, do you want to train again? (yes/no): y
Training...
Saved the trained models.
(If verbose, should show the cross validation scores for each model) 
Best of luck!
```
Provide a test.csv file to predict. 

```
Usage: 
    python3 main.py --test test.csv 

Predicting...
Saved the predictions in submission.csv. 
Best of luck!
```

Other options:
```
-v, --verbose           Verbose. Print the score of each model in the ensemble/bagging. 
--config <config file>  Config file. Specifies ensemble/bagging, selected feature engineering and models. 
                        This defaults to ensemble.cfg. 
```

## Output 

Train: Save the trained models to their respective folders (pickle, SavedModel, h5...), with the same name as the folder. 

Test: Save the ready-to-use predictions to submission.csv in the root folder. 

## Dependencies

- argparse
- configparser
- sklearn
- XGBoost
- LightGBM 

## Improvements

- Add ensemble and bagging of models 