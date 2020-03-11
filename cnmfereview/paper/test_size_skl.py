# tpot best model
import cnmfereview as cr
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import joblib
import autosklearn

from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='path to temp files', type=str)
    parser.add_argument('-s', '--srcdir', help='path to source directory files', type=str)
    args = parser.parse_args()

    SRC_PATH = Path(args.srcdir)
    TMP_PATH = Path(args.dir)
    DATA_PATH = TMP_PATH / 'data'


    num = 15
    n_samples = np.logspace(2,4, num=num, dtype=int)
    indices = range(num)

    accuracies = pd.DataFrame(index=indices,
                            columns=[
                                'Training Size',
                                'Accuracy',
                                'F1'
                            ])

    for i, n in enumerate(n_samples):
        split = 1 - (n/ 14504)
        data = cr.set_up_remote_job(DATA_PATH, feature='combined', split=split)
        training_features, testing_features, training_target, testing_target = data
        automl = joblib.load(f'{SRC_PATH}/models/autosklearn/best_models/autosklearn_model-f.combined-2020-02-25T16:09.joblib')
        automl.refit(training_features.copy(), training_target.copy())

        results = automl.predict(testing_features)        
        acc = accuracy_score(testing_target, results)
        f1 = f1_score(testing_target, results)
        accuracies.iloc[i] = [n, # training size 
                            acc, f1]
        
        print(f"Train set size {n}, \
            Accuracy {acc}, F1 {f1}")
        
    accuracies.to_csv(f'{SRC_PATH}/results-testsize-autoskl.csv')
        
