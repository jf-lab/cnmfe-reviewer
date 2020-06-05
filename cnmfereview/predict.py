import numpy as np
from scipy.io import loadmat, savemat
import config as cfg


def make_predictions(fname, model):
    """
    Predict labels for CNMF-E detected ROIs using a trained
    autosklearn classifier.

    Args:
        fname (str): .mat file containing spatial footprints and 
            traces from CNMF-E (must have ['C', 'C_raw', 'A'] objs)
        model (autosklearn.classifier): an instance of the trained 
            Autosklearn classifier model
    Output:
        predictions (np.ndarray): labels for each ROI
            where 0 (exclude) and 1 (include)
    """
    # load in .mat file with ROIs from experiment
    data = loadmat(fname)
    # spatial downsample factor
    ds = data['ssub']
    max_trace = cfg.max_trace_len
    xpix = cfg.image_shape['x']
    ypix = cfg.image_shape['y']

    trace = data['C_raw'][:, max_trace]]

    spatial = data['A'].reshape(:, int(ypix/ds), int(xpix/ds))
    data_A.append(A)

    model.predict(train)


    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)

def apply_labels(fname, predictions):
    """
    Applies AutoML predicted labels onto the resultant
    .mat file that CNMF-E outputs, and generates a new 
    file with only the "kept" ROIs, with "_automl.mat" 
    appended to the source filename.

    Args:
        fname ([str]): path to .mat file with ROIs
            detected by CNMF-e
        predictions ([np.ndarray, str]): an array or path
            to an array stored in a .npy file of the label 
            predictions for fname from the classifier

    Output:
        returns None
    """
    from scipy.io import loadmat, savemat

    # generate save name and load data
    sname = fname.replace('.mat', '_automl.mat')
    data = loadmat(fname)

    # read in the sklearn predictions
    # 1 = keep, 0 = exclude
    # 1d array of indices to keep
    try:
        keep = predictions.nonzero().ravel()
    except AttributeError:
        predictions = np.load(predictions, allow_pickle=Ture)
        keep = predictions.nonzero().ravel()
        

    # apply changes to the .mat file
    data['A'] = data['A'][:, keep]
    data['C_raw'] = data['C_raw'][keep, :]
    data['C'] = data['C'][keep, :]
    data['S'] = data['S'][keep, :]

    savemat(sname, data, do_compression=True)
