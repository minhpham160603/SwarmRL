# Installation on Ubuntu

This installation procedure has been tested with Ubuntu 22.04, with python 3.10.12

## *Pip* installation

- Install *Pip*:

```bash
sudo apt update
sudo apt install python3-pip 
```

## Virtual environment tools

The safe way to work with *Python* is to create a virtual environment around the project.

For that, you have to install some tools:

```bash
sudo apt install virtualenvwrapper
```
## Install this *SwarmRL* repository

- To install this git repository, go to the directory you want to work in (for example: *~/code/*).

- Git-clone the code of [*SwarmRL*](https://github.com/minhpham160603/SwarmRL):

```bash
git clone https://github.com/minhpham160603/SwarmRL.git
```
This command will create the directory *SwarmRL* with all the code inside it.

- Create your virtual environment. This command will create a directory *env* where all dependencies will be installed:

```bash
cd SwarmRL
python3 -m venv env
```

- To use this newly create virtual environment, as each time you need it, use the command:

```bash
source env/bin/activate
```

To deactivate this virtual environment, simply type: `deactivate`

- With this virtual environment activated, we can install all the dependency with the command:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- Then install the SwarmRL module

```bash
python -m pip install -e ./
```

- To test, you can launch:

```bash
python src/demos/demo_single_env.py
```