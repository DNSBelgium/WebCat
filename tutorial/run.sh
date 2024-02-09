#!/bin/bash

set -e

# Check the existence of a few files to make sure the working directory is correct.
if [ ! -f preprocess.py ] || [ ! -f model.py ] || [ ! -f model_results.py ]
then
  echo Please run the tutorial with the WebCat root as working directory.
  exit 1
fi

if ! which python &> /dev/null
then
  echo Python not found! Please follow the Setup section of the README and make sure your virtual environment is activated.
  exit 1
fi

# Check for a few dependencies that they are installed
if ! pip freeze | grep transformers &> /dev/null || ! pip freeze | grep h5py &> /dev/null
then
  echo Please make sure that the dependencies from requirements.txt are installed.
  exit 1
fi

python preprocess.py train --split 0.25 tutorial/training-minimal-x.parquet tutorial/training-minimal-y.parquet tutorial/training-preprocessed.hdf5

python model.py train --batch-size 4 --seed 42 --epochs 3 tutorial/training-preprocessed.hdf5 tutorial/trained.model

python preprocess.py predict tutorial/testing-minimal-x.parquet tutorial/trained.model tutorial/test-preprocessed.hdf5

python model.py predict tutorial/test-preprocessed.hdf5 tutorial/trained.model tutorial/test-predictions.parquet

python model_results.py --no-details tutorial/testing-minimal-y.parquet tutorial/test-predictions.parquet
