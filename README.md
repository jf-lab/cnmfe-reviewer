# CNMF-E Reviewer

Source code for the paper:

**Automated curation of CNMF-E-extracted ROI spatial footprints and calcium traces using open-source AutoML tools**

Tran LM, Mocle AJ, Ramsaran AI, Jacob AD, Frankland PW, Josselyn SA. (2020) In preparation.

## Getting Started

The [tutorial jupyter notebook](https://github.com/jf-lab/cnmfe-reviewer/blob/master/notebooks/Tutorial.ipynb) in `notebooks/` has information about the tool, and how to how to use it for your own data after you've installed everything. You can make a copy of the notebook and the config.py file and configure the project for your own data. 

### Download example dataset

Most of necessary files are included in the repo, however the processed spatial footprints and traces in the ground truth dataset which can be used in the tutorial can be [downloaded from here](https://drive.google.com/drive/folders/1pGGwUzSI7Hm6gBrilP1SIm0C5bnX7MSO?usp=sharing) (due to GitHub file size limits). Place them in the cnmfe-reviewer/data folder.

### Software requirements
Note: The main AutoML tool used, Autosklearn, works best in Linux environments.
1. Python 3.5+ (ideally with Anaconda)
2. Clone or download this repository onto a local folder on your computer.
3. Change to the directory where you downloaded `cnmfe-reviewer`.
    `cd /path/to/cnfme-reviewer`
4. Create a dedicated conda environment for use of this package (highly recommended to prevent versioning issues with packages you already have installed). Instructions below, but for more detailed instructions, [click here for the conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

```python
# replace myenv_name with your environment name (e.g. cnmfereview)
conda create -n myenv_name python=3.6

# when conda asks to proceed, type 'y'
# this puts your environment in a default folder under the hood

# activate the environment
source activate myenv_name
```

5. While in the conda environment, follow the instructions to [install AutoSklearn](https://automl.github.io/auto-sklearn/master/installation.html#system-requirements).
Namely:
Make sure you have these installed:
- C++ compiler (with C++11 supports) ([get gcc here](https://www.tutorialspoint.com/How-to-Install-Cplusplus-Compiler-on-Linux))
- SWIG (version 3.0 or later) ([get SWIG here](http://www.swig.org/survey.html))

```bash
# install dependancies
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

# (ON LINUX ONLY, skip on MacOS) get the correct compilers and swig
conda install gxx_linux-64 gcc_linux-64 swig

# install auto-sklearn
pip install 'auto-sklearn==0.6.0'
```

6. Finally, install `cnfmereview`:

```bash
# install cnmfereview (you should be in the cnmfe-reviewer directory)
pip install ./
```

7. To make the environment available in Jupyter Notebooks:
```bash
conda install ipykernel
ipython kernel install --user --name=<any_name_for_kernel>
```

8. When you're done, you can deactivate the environment:
```bash
conda deactivate
```
