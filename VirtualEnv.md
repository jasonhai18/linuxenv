With virtualenvwrapper (user friendly wrappers for the functionality of virtualenv)

## Install virtualenv
Install virtualenv with

	sudo apt-get install virtualenv

(for Ubuntu 14.04 (trusty) install python-virtualenv)

## Install virtualenvwrapper
The reason we are also installing virtualenvwrapper is because it offers nice and simple commands to manage your virtual environments. There are two ways to install virtualenvwrapper:

### As Ubuntu package (from Ubuntu 16.04)
Run

	sudo apt install virtualenvwrapper

then run

	echo "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh" >> ~/.bashrc
### Using pip
>####Install and/or update pip

>Install pip for Python 2 with

	sudo apt-get install python-pip
>or for Python 3

	sudo apt-get install python3-pip
>(if you use Python 3, you may need to use pip3 instead of pip in the rest of this guide).

>Optional (but recommended): Turn on bash autocomplete for pip

>Run

	pip completion --bash >> ~/.bashrc
>and run 
	
	source ~/.bashrc 

>to enable.

>#### Install virtualenvwrapper

>Because we want to avoid sudo pip we install virtualenvwrapper locally (by default under ~/.local) with:

	pip install --user virtualenvwrapper

>and

	echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
>#### Source virtualenvwrapper in .bashrc

	echo "source ~/.local/bin/virtualenvwrapper.sh" >> ~/.bashrc
### Setup virtualenv and virtualenvwrapper:
First we export the WORKON_HOME variable which contains the directory in which our virtual environments are to be stored. Let's make this ~/.virtualenvs

	export WORKON_HOME=~/.virtualenvs

now also create this directory

	mkdir $WORKON_HOME
and put this export in our ~/.bashrc file so this variable gets automatically defined

	echo "export WORKON_HOME=$WORKON_HOME" >> ~/.bashrc
We can also add some extra tricks like the following, which makes sure that if pip creates an extra virtual environment, it is also placed in our WORKON_HOME directory:

	echo "export PIP_VIRTUALENV_BASE=$WORKON_HOME" >> ~/.bashrc 
#### Source ~/.bashrc to load the changes

	source ~/.bashrc
#### Test if it works

Now we create our first virtual environment. The -p argument is optional, it is used to set the Python version to use; it can also be python3 for example.

	mkvirtualenv -p python2.7 test
You will see that the environment will be set up, and your prompt now includes the name of your active environment in parentheses. Also if you now run

	python -c "import sys; print sys.path"
you should see a lot of /home/user/.virtualenv/... because it now doesn't use your system site-packages.

You can deactivate your environment by running

	deactivate
and if you want to work on it again, simply type

	workon test
Finally, if you want to delete your environment, type

	rmvirtualenv test
Enjoy!
