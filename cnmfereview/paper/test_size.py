# tpot best model
import cnmfereview as cr
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from tpot.export_utils import set_param_recursive
from sklearn import svm, naive_bayes, neural_network
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

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

        exported_pipeline = make_pipeline(
            VarianceThreshold(threshold=0.0001),
            SGDClassifier(alpha=0.001, eta0=0.01, fit_intercept=True, l1_ratio=1.0, 
                learning_rate="constant", loss="modified_huber", penalty="elasticnet",
                power_t=10.0)
        )
        # Fix random state for all the steps in exported pipeline
        set_param_recursive(exported_pipeline.steps, 'random_state', 23)
        
        print("Training pipeline.")
        exported_pipeline.fit(training_features, training_target)
        results = exported_pipeline.predict(testing_features)    
        acc = accuracy_score(testing_target, results)
        f1 = f1_score(testing_target, results)
        accuracies.iloc[i] = [n, # training size 
                            acc, f1]
        
        print(f"Train set size {n}, \
            Accuracy {acc}, F1 {f1}")
        
    accuracies.to_csv(f'{SRC_PATH}/results-testsize.csv')
        
