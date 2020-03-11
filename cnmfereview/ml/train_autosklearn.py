import os
import pickle
from datetime import datetime
from pathlib import Path
from shutil import copyfile
import numpy as np
import pandas as pd
import autosklearn.metrics
import autosklearn.classification
import sklearn.metrics
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.externals import joblib
import argparse

cv_folds = 5 # CV is by default stratifiedKFold for auto-sklearn
random_state = 23 # 162000s is 45 hours # how long to run auto-ml in seconds (default = 3600)
#max_time_secs = 162000 #86400 # how many seconds for a single call to a model (fitting/eval) (default = 360)
max_eval_time_secs = 10800 
population_size = 100


# weight precision (FP) > recall (FN) 
def get_f1_beta(y_true, y_pred):
    return sklearn.metrics.fbeta_score(y_true, y_pred,0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', help='amount of time to run autosklearn in days', type=float)
    parser.add_argument('-d', '--dir', help='path to temp files', type=str)
    parser.add_argument('-f', '--feats', 
                        help='which feature set to use [spatial, trace, combined]',
                        type=str, default='combined')
    parser.add_argument('-s', '--srcdir', help='path to source directory files', type=str)
    args = parser.parse_args()

    SRC_PATH = Path(args.srcdir)
    TMP_PATH = Path(args.dir)
    DATA_PATH = TMP_PATH / 'data'
    MODELS_PATH = SRC_PATH / 'models' 
    MODELS_PATH.mkdir(exist_ok=True)

    AUTOSKLEARN_PATH = Path(MODELS_PATH / 'autosklearn')
    AUTOSKLEARN_PATH.mkdir(exist_ok=True)
    AUTOSKLEARN_TOPMODELS = AUTOSKLEARN_PATH / 'best_models'
    AUTOSKLEARN_TOPMODELS.mkdir(exist_ok=True)
    AUTOSKLEARN_ENSEMBLES = AUTOSKLEARN_PATH / 'ensembles'
    AUTOSKLEARN_ENSEMBLES.mkdir(exist_ok=True)
    runscripts_dir = AUTOSKLEARN_PATH / 'run_scripts'
    runscripts_dir.mkdir(exist_ok=True)
    model_logs_path = TMP_PATH / 'model_logs'
    model_logs_path.mkdir(exist_ok=True)
    RESULTS_PATH = SRC_PATH / 'results'
    RESULTS_PATH.mkdir(exist_ok=True)

    f1beta = autosklearn.metrics.make_scorer(name='f1beta', score_func=get_f1_beta)

    max_days = args.time
    feats = args.feats
    print(f'feats are: {args.feats}.csv')

    max_time_secs = (max_days * 24 * 60 * 60) - 3600 # subtract an  hour to make sure that final wrap up of script can still finish with given time alloted on SCC
    max_time_secs = int(max_time_secs)

    START_TIME = datetime.now().isoformat(timespec='minutes')
    print(f'Start time is: {START_TIME}')

    autosklearn_name = 'autosklearn_exported_pipeline.feats.' + str(feats) + '.time.' + str(START_TIME) + '.py'
    run_log = 'run_' + autosklearn_name

    print("autosklearn runscript written to:", runscripts_dir / run_log)
    copyfile(os.path.realpath(__file__), runscripts_dir / run_log)

    # Read in data (spatial, temporal, features)
    import cnmfereview as cr
    X_train, X_test, y_train, y_test = cr.set_up_remote_job(DATA_PATH, feature=feats)
    from skimage.transform import downscale_local_mean
    X_train = downscale_local_mean(X_train, (1,4))
    X_test = downscale_local_mean(X_test, (1,4))
    # stratified K fold chooses same proportion of labels per fold
    kf = model_selection.RepeatedStratifiedKFold(n_splits=cv_folds,
                                                n_repeats=5, 
                                                random_state=random_state)


    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=max_time_secs,
        per_run_time_limit=max_eval_time_secs,
        ensemble_size = population_size,
        n_jobs=16,
        ml_memory_limit=10000,
        ensemble_memory_limit=5000,
        tmp_folder= model_logs_path / f'autosklearn_{feats}_{START_TIME}', # folder to store configuration output and log files
        output_folder = AUTOSKLEARN_TOPMODELS / f'autosklearn_out_{feats}_{START_TIME}', # folder to store predictions for optional test set
        delete_tmp_folder_after_terminate=False,
        resampling_strategy=model_selection.RepeatedStratifiedKFold,
	resampling_strategy_arguments={'folds': cv_folds, 'n_repeats': 5, 'random_state': random_state})

    # fit() changes the data in place, but refit needs the original data. We
    # therefore copy the data. In practice, one should reload the data
    #score_function = autosklearn.metrics.f1
    score_function = f1beta

    automl.fit(X_train.copy(), y_train.copy(), metric=score_function)
    # During fit(), models are fit on individual cross-validation folds. To use
    # all available data, we call refit() which trains all models in the
    # final ensemble on the whole dataset.
    automl.refit(X_train.copy(), y_train.copy())

    print(automl.sprint_statistics())

    joblib.dump(automl, AUTOSKLEARN_TOPMODELS / f'autosklearn_model-f.{feats}-{START_TIME}.joblib')
    
    # dump out the ensemble model as a pickled dictionary
    x = automl.show_models()
    results = {'ensemble': x}
    pickle.dump(results, open(AUTOSKLEARN_ENSEMBLES / f'ensemble_model_description-f.{feats}-{START_TIME}.pickle', 'wb'))

    predictions = automl.predict(X_test)
    f1 = f1_score(y_test, predictions)
    print('f1:', f1)
    final_result = np.column_stack([y_test, predictions])
    final_result = pd.DataFrame(final_result, columns=['True Labels', 'Predictions'])
    final_result.to_csv(RESULTS_PATH / f'autosklearn-f.{feats}-{START_TIME}.csv', index=None)
