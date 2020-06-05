import numpy as np
from scipy.io import loadmat, savemat

def make_predictions(fname, model, cfg):
    """
    Predict labels for CNMF-E detected ROIs using a trained
    autosklearn classifier.

    Args:
        fname (str): .mat file containing spatial footprints and
            traces from CNMF-E (must have ['C', 'C_raw', 'A'] objs)
        model (autosklearn.classifier): an instance of the trained
            Autosklearn classifier model
        cfg (dict): dictionary of configuration settings
    Output:
        predictions (np.ndarray): labels for each ROI
            where 0 (exclude) and 1 (include)
    """
    # load in .mat file with ROIs from experiment
    data = loadmat(fname)

    # spatial downsample factor
    ds = data['ssub']
    max_trace = cfg.max_trace_len
    xpix = int(cfg.img_shape['x']/ds)
    ypix = int(cfg.img_shape['y']/ds)

    trace = data['C_raw'][:, max_trace]

    spatial = data['A'].reshape(
        (len(data['A']), ypix, xpix)
        )
    spatial = crop_footprint(spatial, cfg.img_crop_size)

    spatial_flattened = np.reshape(
        self.spatial, [
            len(self.spatial),
            self.spatial.shape[1]*self.spatial.shape[2]
            ]
        )

    self.combined = np.concatenate(
        (spatial_flattened, self.trace),
        axis=1


    model.predict(train)

    predictions = model.predict(X_test)

    f1 = f1_score(y_test, predictions)

