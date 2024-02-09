# WebCat tutorial

This directory consists of a minimal dataset needed to train a model, and a `run.sh` script that runs all necessary commands to train and evaluate a model using this minimal dataset (following the instructions from the project README).
The performance will not be high, but it demonstrates the working of WebCat.
The commands are configured such that training should not take long and can be done on a CPU (so no GPU needed).

`run.sh` should be executed in the WebCat root as working directory (so the directory above this `tutorial` directory), with the correct Python virtual environment activated.
This means that you should have done the "Setup" section from WebCat's README.
The other training and testing steps are performed by the tutorial script.