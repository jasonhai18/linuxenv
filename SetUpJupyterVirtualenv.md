# Getting started with JupyterLab
## conda
If you use conda, you can install it with:
  conda install -c conda-forge jupyterlab
## pip
If you use pip, you can install it with:
  pip install jupyterlab
If installing using pip install --user, you must add the user-level bin directory to your PATH environment variable in order to launch jupyter lab.

# Getting started with the classic Jupyter Notebook
## Prerequisite: Python
While Jupyter runs code in many programming languages, Python is a requirement (Python 3.3 or greater, or Python 2.7) for installing the JupyterLab or the classic Jupyter Notebook.

## Installing Jupyter Notebook using Anaconda
We strongly recommend installing Python and Jupyter using the Anaconda Distribution, which includes Python, the Jupyter Notebook, and other commonly used packages for scientific computing and data science.

First, download Anaconda. We recommend downloading Anaconda’s latest Python 3 version.

Second, install the version of Anaconda which you downloaded, following the instructions on the download page.

Congratulations, you have installed Jupyter Notebook! To run the notebook, run the following command at the Terminal (Mac/Linux) or Command Prompt (Windows):

  jupyter notebook
See Running the Notebook for more details.

## Installing Jupyter Notebook with pip
As an existing or experienced Python user, you may wish to install Jupyter using Python’s package manager, pip, instead of Anaconda.

If you have Python 3 installed (which is recommended):

  python3 -m pip install --upgrade pip
  python3 -m pip install jupyter
If you have Python 2 installed:

python -m pip install --upgrade pip
python -m pip install jupyter
Congratulations, you have installed Jupyter Notebook! To run the notebook, run the following command at the Terminal (Mac/Linux) or Command Prompt (Windows):

  jupyter notebook
# Setting up the virtual environment for Jupyter
## Find the Virtualenv pwd
Go to
    cd /.virtualenvs/
to find how many virtualenvs you create. Then go to
  ~/.virtualenvs/tf-gpu/bin
to find the python file(python is the default configuration language in your virtualenv, not python3 or python2).
  pwd
get the path.
## Find the jupyter notebook kernel folder
Go to search 
  sudo find . -name "kernel.json"
and it should be in the path
  ~/.local/share/jupyter/kernels
ls to find how many kernels do you have.

Then 
  mkdir
make the new folder which has the same name as your virtualenv.
  touch kernel.json
and copy the code from the initial kernel (~/.local/share/jupyter/kernels/python3).

And change the path to your virtualenv path.
## Install jupyter in your virtualenv
  pip install jupyter
You must sudo install the jupyter notebook in your boot environment.
