import os
from datetime import datetime
from sklearn import metrics
from sklearn import model_selection
import numpy as np
from sklearn.metrics import f1_score
from tpot import TPOTClassifier
import sys
#import config
from pathlib import Path
from shutil import copyfile
import argparse

cv_folds = 10
random_state = 23
max_eval_time_mins = 5  # how many minutes is a single pipeline allowed
population_size = 100 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', help='amount of time to run autosklearn in days', type=float)
    parser.add_argument('-d', '--dir', help='path to temp files', type=str)
    parser.add_argument('-f', '--feat', 
                        help='which feature set to use [spatial, trace, combined]',
                        type=str, default='combined')
    parser.add_argument('-s', '--srcdir', help='path to source directory files', type=str)
    args = parser.parse_args()
    
    SRC_PATH = Path(args.srcdir)
    TMP_PATH = Path(args.dir)
    DATA_PATH = TMP_PATH / 'data'
    MODELS_PATH = SRC_PATH / 'models' 
    MODELS_PATH.mkdir(exist_ok=True)
    TPOT_PATH = Path(MODELS_PATH / 'tpot')
    TPOT_PATH.mkdir(exist_ok=True)
    TPOT_TOPMODELS = TPOT_PATH / 'best_models'
    TPOT_TOPMODELS.mkdir(exist_ok=True)
    runscripts_dir = TPOT_PATH / 'run_scripts'
    runscripts_dir.mkdir(exist_ok=True)
    checkpoint_path = TPOT_PATH / 'checkpoint_models'
    checkpoint_path.mkdir(exist_ok=True)
    RESULTS_PATH = SRC_PATH / 'results' 
    RESULTS_PATH.mkdir(exist_ok=True)

    n_jobs = 16

    max_time_in_days = args.time
    max_time_mins = (max_time_in_days * 24 * 60) - 60
    max_time_mins = int(max_time_mins)

    START_TIME = datetime.now().isoformat(timespec='minutes')
    print(f'Start time is: {START_TIME}')
    tpot_name = f'exported_pipeline.time.{START_TIME}.{args.feat}.tpot.py'
    run_log = 'run_' + tpot_name

    print("TPOT runscript written to:", runscripts_dir / run_log)
    copyfile(os.path.realpath(__file__), runscripts_dir / run_log)

    # Read in data (spatial, temporal, features)
    import cnmfereview as cr
    X_train, X_test, y_train, y_test = cr.set_up_remote_job(DATA_PATH, feature=args.feat) 
    
    # stratified K fold chooses same proportion of labels per fold
    kf = model_selection.RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=5, random_state=random_state)

    config_dict = None 
    tpot = TPOTClassifier(population_size=population_size, verbosity=2, scoring='f1',
                          random_state=random_state, cv=kf, n_jobs=n_jobs,
                          max_time_mins=max_time_mins, max_eval_time_mins=max_eval_time_mins,
                          config_dict=config_dict, memory=None, periodic_checkpoint_folder=checkpoint_path)
    print(f'Starting TPOT training at: {START_TIME}')
    tpot.fit(X_train, y_train)

    tpot.export(TPOT_TOPMODELS / tpot_name)

    predictions = tpot.predict(X_test)
    np.savetxt(RESULTS_PATH / f'tpot_preds-{START_TIME}-{args.feat}.csv', predictions, delimiter=',')

