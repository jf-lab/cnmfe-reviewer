from setuptools import setup

setup(name="cnmfereview",
      version='0.1',
      description="Using automl tools to review for CNMF-E extracted cells",
      author="linamnt",
      license="MIT",
      packages=['cnmfereview'],
      install_requires=[
          'auto-sklearn>=0.6.0',
          'scikit-learn>0.21.0,<0.22.0'
          ],
      zip_safe=False)
