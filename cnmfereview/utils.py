import numpy as np
import os
from pathlib import Path


class UnlabeledDataset(object):
    def __init__(
        self,
        mat_file: str,
        img_shape: dict,
        img_crop_size: dict,
        max_trace: int,
    ):
        """
        Class to prepare unlabeled cnmf-e output ROIs for classification.

        Args:
            mat_file (str): path to .mat file containing ['C_raw', 'A']
                objects (trace and spatial footprints).
            img_shape (dict): A dictionary with keys, "x" and "y" with the
                x and y dimensions in pixels of your input spatial images
                (i.e. the dimensions of a frame of your video).
            img_crop_size (dict): A dictionary with keys, "x" and y" with
                the x and y dimensions in pixels of the cropped image from the
                total field of view that contains only an individual ROIs.
            max_trace (int): Integer value of the number of frames to take from
                each trace to use in the dataset.
        """
        from scipy.io import loadmat, savemat
        self.data = loadmat(mat_file)

        self.mat_file = mat_file

        ds = self.data['ssub']      # spatial downsample factor

        self.xpix = int(img_shape['x']/ds)
        self.ypix = int(img_shape['y']/ds)
        self.max_trace = max_trace
        self.img_crop_size = img_crop_size

        print(f"Sucessfully loaded data. There are {self.data['C_raw'].shape[0]} ROIs \
               in this .mat file.")

        self.preprocess_dataset()


    def preprocess_dataset(self):
        """
        Prepare the ROI data in the .mat for classification.
        """
        trace = self.data['C_raw']
        trace = align_traces(trace)
        trace = trace_window(trace, self.max_trace)
        self.trace = normalize_traces(trace)

        spatial = self.data['A'].transpose(1,0)
        spatial = spatial.reshape((
            self.trace.shape[0], self.ypix, self.xpix
            ))

        self.spatial = crop_footprint(spatial, self.img_crop_size)

        spatial_flattened = np.reshape(
            self.spatial, [
                len(self.spatial),
                self.spatial.shape[1]*self.spatial.shape[2]
            ])

        self.combined = np.concatenate(
            (spatial_flattened, self.trace),
            axis=1)

        print(f"Spatial footprints cropped, and traces normalized and cropped.")


    def apply_labels(self, predictions):
        """
        Applies AutoML predicted labels onto the resultant
        .mat file that CNMF-E outputs, and generates a new
        file with only the "kept" ROIs, with "_automl.mat"
        appended to the source filename.

        Args:
            predictions ([np.ndarray, str]): an array or path
                to an array stored in a .npy file of the label
                predictions for fname from the classifier

        Output:
            returns None
        """
        from scipy.io import loadmat, savemat

        # generate save name and load data
        sname = self.mat_file.replace('.mat', '_automl.mat')

        # read in the sklearn predictions
        # 1 = keep, 0 = exclude
        # 1d array of indices to keep
        keep = predictions.nonzero()[0]

        # add the labels to the original .mat file
        self.data['keep'] = keep
        savemat(self.mat_file, self.data, do_compression=True)

        # apply changes to a new copy of the .mat file
        self.data['A'] = self.data['A'][:, keep] # footprints
        self.data['C_raw'] = self.data['C_raw'][keep, :] # traces (raw)
        self.data['C'] = self.data['C'][keep, :] # traces (deconvolved)
        self.data['S'] = self.data['S'][keep, :] # inferred spikes

        savemat(sname, self.data, do_compression=True)
        print(f"Saved updated arrays with only positive labels in {sname}")


class Dataset(object):
    """
    Class to generate dataset containing ROI spatial
    footprints and traces, and utility functions
    to process and load the data specifically for
    training your custom classifiers.
    """
    def __init__(self,
                 data_paths: dict,
                 exp_id: str,
                 img_shape: dict,
                 img_crop_size: dict,
                 max_trace: int,
                 preprocess=False):
        """
        Initialize an instance of the Dataset class.

        Args:
            data_paths (dict): A dictionary containing paths and filenames
                including "data_directory" which is the path to the data
                where all the spatial, trace and targets files should be,
                "spatial" which is the .npy filename of the spatial footprints,
                "trace" which is the .npy filename of the raw calcium traces,
                "targets" which is the .npy filename of your manually reviewed
                labels for the spatial and trace data.
            exp_id (str): Name for the classifier.
            img_shape (dict): A dictionary with keys, "x" and "y" with the
                x and y dimensions in pixels of your input spatial images
                (i.e. the dimensions of a frame of your video).
            img_crop_size (dict): A dictionary with keys, "x" and y" with
                the x and y dimensions in pixels of the cropped image from the
                total field of view that contains only an individual ROIs.
            max_trace (int): Integer value of the number of frames to take from
                each trace to use in the dataset.
            preprocess (bool): If True, preprocess the raw data even if
                processed data already exists. This will overwrite the
                "A_cropped.npy" and "C_normalized.npy" in the data directory.
                This may be useful if you've updated your raw data and would
                like to preprocess again.
        """
        self.DATADIR = Path(data_paths['data_directory'])
        self.img_crop_size = img_crop_size
        self.max_trace = max_trace
        self.dtype = np.float32
        self.exp_id = exp_id

        preprocess_spatial = (not os.path.exists(
            self.DATADIR / (self.exp_id + "A_cropped.npy"))) or preprocess
        preprocess_trace = (not os.path.exists(
            self.DATADIR / (self.exp_id + "Craw_normalized.npy"))) or preprocess

        if preprocess_spatial:
            # crop spatial footprints
            spatial = np.load(self.DATADIR / data_paths['spatial'],
                              allow_pickle=True).astype(self.dtype)
            
            self.spatial = crop_footprint(spatial, crop_dim=img_crop_size)
            
            np.save(self.DATADIR / (self.exp_id + "A_cropped.npy"), 
                    self.spatial)
            del spatial  # clear up memory
            
            print(f"Spatial data preprocessed, \
                saved in {self.DATADIR / (self.exp_id + 'A_cropped.npy')}")
        else:
            self.spatial = np.load(self.DATADIR / (self.exp_id + 'A_cropped.npy'),
                                   allow_pickle=True).astype(self.dtype)
            print("No preprocessing on spatial data")
            print(f"File {self.DATADIR / (self.exp_id + 'A_cropped.npy')} already exists and has been loaded instead.")

        if preprocess_trace:
            # crop and normalize traces
            trace = np.load(self.DATADIR / data_paths['trace'],
                            allow_pickle=True)
            trace = process_traces(trace, self.max_trace)
            self.trace = trace            
            np.save(self.DATADIR / (self.exp_id + 'Craw_normalized.npy'), self.trace)
            del trace # clear up memory
            
            print(f"Trace data preprocessed, saved in \
            {self.DATADIR /( self.exp_id + 'Craw_normalized.npy')}")
        else:
            self.trace = np.load(self.DATADIR / (self.exp_id + 'Craw_normalized.npy'),
                             allow_pickle=True)
            print(f"No preprocessing on trace data. \
                  {self.DATADIR / (self.exp_id + 'Craw_normalized.npy')} already \
                  exists and has been loaded instead.")

        self.targets = np.load(self.DATADIR / data_paths['targets']
                               ).astype(self.dtype)

        assert (len(self.targets) == len(self.spatial) and
                len(self.targets == len(self.trace))),\
            f"Number of targets labels, {self.targets.shape} does not match \
            number of ROIs in both spatial, {len(self.spatial)} and \
            trace, {len(self.trace)}."

        # for flattening and concatenating the data
        spatial_flattened = np.reshape(
            self.spatial, [
                len(self.spatial),
                self.spatial.shape[1]*self.spatial.shape[2]
                ]
            )

        self.combined = np.concatenate(
            (spatial_flattened, self.trace),
            axis=1
            )

        print("Successfully loaded data.")
        

    def split_training_test_data(self, test_split: float=0.2,
                                seed: int=0):
        """
        Splits spatial footprints, traces and targets into
        training, test sets with sizes based on test_split.
        Uses a StratifiedShuffleSplit to assure both the train
        and test sets have the same proportion of negative labels.
        The spatial footprints are flattened, trace data is
        concatenated to the end.

        Args:
            test_split (float, optional): Proportion of samples
                to be set aside and used in the test set. Defaults to 0.2
            seed (int, optional): Random seed used for splitting data.
                If you do not want the results to be deterministic,
                set to None. Otherwise, you can simply
                set a different seed to get a different split.
                Defaults to 0.
        Returns:
            x_train, x_test, y_train, y_test

        """
        data = self.combined
        targets = self.targets

        # train/test split
        # stratifies the split so training and test set have same
        # proportion of positive and negative labels
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=1,
                                    test_size=test_split,
                                    random_state=seed)

        for train_index, test_index in sss.split(data, targets):
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = targets[train_index], targets[test_index]
            print("Training and test data loaded")

        return x_train, x_test, y_train, y_test



def crop_image(img, peak, x_margin, y_margin):
    """
    Crops an image centered on the peak coordinates, with a
    margin of x_margin and y_margin on both sides, in both
    dimensions.

    The resultant image will be of shape:
    [x_margin*2, y_margin*2]

    Args:
        img (np.ndarray): 2-d array of the image to crop
        peak (float): [description]
        x_margin (int): margin around both sides of peak in x-axis
        y_margin (int): margin around both sides of peak in y-axis

    Returns:
        new_img (np.ndarray): cropped image
    """
    new_img = np.zeros((int(x_margin*2), int(y_margin*2)))
    img = img[peak[0]-x_margin:peak[0]+x_margin,
              peak[1]-y_margin:peak[1]+y_margin]
    new_img[:img.shape[0], :img.shape[1]] = img

    return new_img


def crop_footprint(spatial: np.ndarray,
                   crop_dim: dict
                  ):
    """
    Crop each spatial footprint based on peak of each ROI.

    Args:
        spatial (np.ndarray): Spatial footprints from whole
            field of view.
        crop_dim (dict): {"x": int, "y": int}, dimensions, in pixels
            of the final cropped image. If only one value provided,
            the final cropped image will be a square of shape
            crop_dim x crop_dim
    Returns:
        cropped_spatail (np.ndarray): Array of cropped spatial
            footprints of shape (spatial[0], crop_dim[0], crop_dim[1])
    """
    if isinstance(crop_dim, int):
        x = crop_dim
        y = crop_dim
    elif isinstance(crop_dim, dict):
        x = crop_dim['x']
        y = crop_dim['y']

    args = spatial.reshape(spatial.shape[0],-1).argmax(-1)
    maxima = np.unravel_index(args, spatial.shape[-2:])
    peak_coor = list(zip(maxima[0], maxima[1]))
    cropped_spatial = np.zeros((spatial.shape[0],
                                x,
                                y))
    for ix, peak in enumerate(peak_coor):
        cropped_spatial[ix] = crop_image(spatial[ix].squeeze(),
                                         peak,
                                         int(x/2),
                                         int(y/2))

    return cropped_spatial


def align_traces(traces, longest=None, pad=True):
    """
    Since all your calcium traces may not be the same
    length, this function will either pad shorter arrays
    with zeros to match the same length as the longest
    traces (if "longest" is set to None), or crop the traces to an
    arbitrary length provided by "longest".

    This is NOT the same crop that will applied to find
    the trace window. This is merely a preprocessing step
    to make sure the traces are aligned and the same length
    to prevent jagged arrays and to make it easier
    to apply vectorized operations.

    Note that "longest" must be an integer smaller than
    the length of the longest trace in the dataset. You may
    want to set your own "longest" if the longest trace in
    your dataset is disproportionately long, and may
    take long to load. Depending on your machine, and how
    many ROIs you have, > 10 000 frames may be too much.


    Args:
        traces (array): calcium traces gathered from CNMF-E
        longest (int, optional): Max length of all traces to
            take.
            Must be  Defaults to None.
        pad (bool, optional): Pad shorter ararys with zeros at
            the end to match default length. Defaults to True.
    """
    def roundup(x):
        """
        Convenience function to round up trace array lengths
        to the nearest 100th
        """
        return x if x % 100 == 0 else x + (100 - x % 100)

    if longest is not None:
        longest_trace = longest
        traces = np.array([x[:longest_trace] for x in traces])
    else:
        longest_trace = roundup(max([i.shape[0] for i in traces]))

    # if pad=True, pad arrays with 0 values to make all the same shape
    if pad:
        traces = np.array(
            [np.pad(x,
                    (0, longest_trace - x.shape[0]),
                    'constant'
                    ) for x in traces])

    return traces


def trace_window(traces, size=500, threshold=0.75):
    """
    Find the first occurrence of activity that reaches a
    threshold larger than 75 percent of the trace range,
    and take a frame window of the correct size centered on it.
    This prevents taking a trace window where there is little
    to no activity.

    Args:
        traces (np.ndarray): Calcium trace data.
        size (int, optional): Number of frames to take from
            the trace. Defaults to 500.
        threshold (float, optional): Threshold of activity in
            the trace range (i.e. from minimum df/f signal to
            max) to centre the trace window on. Defaults to 0.75.
    Returns:
        new_traces: Modified traces of length, size, centered
            on the first occurence of activity passing the
            threshold.
    """
    trace_min = traces.min(axis=1)
    trace_max = traces.max(axis=1)
    trace_mid = ((
        traces.max(axis=1) - traces.min(axis=1)
        )*threshold + trace_min)

    # initialize empty array
    new_traces = np.zeros((traces.shape[0], size))
    for ix, i in enumerate(trace_mid):
        # find the first occurence of calcium activity beyond baseline
        centre_idx = np.argmax(traces[ix] > i)
        start_idx = centre_idx - int(size/2) if centre_idx > int(size/2) else 0
        if start_idx > (traces.shape[1] - size):
            start_idx = traces.shape[1] - size
        new_traces[ix] = traces[ix, start_idx:start_idx+size]

    return new_traces

                  
def normalize_traces(traces):
    """
    Normalize and scale calcium trace data so all
    traces range between 0 and 1.

    Args:
        traces (np.ndarray): Raw calcium trace data.
    Returns:
        traces_norm (np.ndarray): Normalized caclium
            trace data.
    """
    peak = traces.max(axis=1)[:, np.newaxis]
    trough = traces.min(axis=1)[:, np.newaxis]
    trace_range = peak - trough

    # to prevent division by zero if there is no trace activity
    # though unlikely
    trace_range[np.where(trace_range == 0)] = 1
    traces_norm = (traces - trough) / trace_range

    return traces_norm


def process_traces(traces, max_trace):
    trace = align_traces(traces)

    trace = normalize_traces(trace)

    trace = trace_window(
        trace,
        size=max_trace,
        threshold=0.75
    )

    return trace


def setup_1p_data(
        dataset: Dataset,
        feature: str='combined',
        split: float=0.2
        ):
    """
    Prepares dataset from raw spatial and trace data to final
    training and test sets.

    Args:
        dataset (Dataset): Dataset object containing your spatial
            trace, and targets data.
        feature (str, optional): Which data types to use.
            - "spatial" only load spatial data and targets
            - "trace" only load trace data and targets
            - "combined" load concatenated spatial and trace
            data with targets
            Defaults to "combined".
        split (float, optional): How to split train and test sets.
            Defaults to 0.2.

    Returns:
        x_train, x_test, y_train, y_test
    """
    # set random seed (sklearn)
    seed = 0

    spatial = dataset.spatial
    trace = dataset.trace
    targets = dataset.targets

    # array shapes need to work with sklearn
    px = spatial.shape[1]*spatial.shape[2]

    spatial = np.reshape(spatial, (len(spatial), px))

    dt = {
        'spatial': spatial,
        'trace': traces_n,
        'combined': np.concatenate((spatial, traces_n), axis=1)
        }

    data = dt[feature]

    # train/test split
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=split, random_state=seed)

    for train_index, test_index in sss.split(data, targets):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

    print("Training and test data loaded")

    return x_train, x_test, y_train, y_test


def preprocess_2p_dataset(
        spatial,
        traces,
        targets,
        methods=None,
        feature='combined', split=0.2
    ):
    seed = 0
    px = spatial.shape[1]
    spatial = np.reshape(spatial, (len(spatial), px*px))
    dt = {'spatial': spatial,
            'trace': traces,
            'combined': np.concatenate((spatial, traces), axis=1)}

    data = dt[feature]

    # train/test split
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=split,
                                random_state=seed)

    for train_index, test_index in sss.split(data, targets):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        print("Training and test data loaded")
        if methods is not None:
            m_train, m_test = methods[train_index], methods[test_index]
            print("Target descriptions loaded \
                   (ground truth dataset only)")
            return x_train, x_test, y_train, y_test, m_train, m_test

        return x_train, x_test, y_train, y_test

def plot_rois(
    dataset: UnlabeledDataset,
    subset: list
):
    """
    Plot the results of classifier on your
    unlabeled data for examination.

    Args:
        dataset (cnmfereview.utils.UnlabeledDataset): UnlabeledDataset
            class object.
        subset (np.ndarray): List of indices of the ROIs you'd like to preview.
    """
    tr = dataset.trace[subset]
    sp = dataset.spatial[subset]

    import matplotlib.pyplot as plt

    cols = 5
    rows = len(subset)//cols + (1 if len(subset)%cols else 0)

    fig = plt.figure(figsize=(10*rows,20))
    ax = fig.subplots(nrows=rows*2, ncols=cols)
    ax = ax.ravel()

    cell_no = 0
    
    for j in range(rows):
        for i in range(cols):
            # if reached the end of cells, stop
            if cell_no >= len(subset):
                break
            
            #top row
            ax_no = j*2*cols+i
            cell_no = j*cols + i
            ax[ax_no].plot(tr[cell_no])
            ax[ax_no].set_ylim(0,1)
            ax[ax_no].set_xlim(0,505)
            ax[ax_no].set_xticks([])
            ax[ax_no].set_yticks([])
            ax[ax_no].title.set_text(f"CellID: {subset[cell_no]}")

            #bottom row
            ax_no += cols
            ax[ax_no].imshow(sp[cell_no])
            ax[ax_no].set_yticks([])
            ax[ax_no].set_xticks([])

            cell_no += 1

    plt.show()



def classification_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray
):
    """
    Create four arrays of the indices of false positives,
    false negatives, true positives and true negatives
    from predicted (y_pred) and target (y_true) classes.

    Args:
        y_true (np.ndarray): Array of the true target values
        y_pred (np.ndarray): Array of predicted values
    """
    def false_positive(y_true, y_pred):
        idx = np.argwhere(y_pred > y_true)
        return idx

    def false_negative(y_true, y_pred):
        idx = np.argwhere(y_true > y_pred)
        return idx
                  
                  
    def true_positive(y_true, y_pred):
        idx = np.argwhere((y_pred == y_true) & (y_true == 1))
        return idx

    def true_negative(y_true, y_pred):
        idx = np.argwhere((y_pred == y_true) & (y_true == 0))
        return idx
                  
    fp = false_positive(y_true, y_pred).ravel()
    fn = false_negative(y_true, y_pred).ravel()
    tp = true_positive(y_true, y_pred).ravel()
    tn = true_negative(y_true, y_pred).ravel()

    return (fp, fn, tp, tn)


def retrieve_sp_tr(data,
                   x: int,
                   y: int):
    """
    Retreives the original spatial footprint and trace data
    from flattened and concatenated array used for classification

    Args:
        data (np.ndarray): NxM array where N is the number of cells,
            and M is the length of the flattened, concatenated
            spatial and trace data.
        x (int, optional): Original cropped image x dimension.
        y (int, optional): Original cropped image y dimension.

    Returns:
        spatial: original spatial images
        tr: original trace data
    """
    split = x*y # index of the end of ROI, start of trace

    spatial = data[:, :split]
    spatial = spatial.reshape((len(spatial), x, y))

    tr = data[:, split:]

    return spatial, tr

