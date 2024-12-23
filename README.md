# Final Project Repo

## Installation Instructions

### Setup conda env.
Activate using
```
conda activate cs7643-final-proj
```

Pytorch isn't included in the conda environment as version might be different for different os/machine.
Can install using:
```
conda install pytorch torchvision -c pytorch
```

Might need to install ipykernel
```
conda install ipykernel --update-deps --force-reinstall
```

### Running on GCP Notebook VM

In order to have tensorboard install in a VM, the following can be used:
```
import sys
!{sys.executable} -m pip install tensorboard
```
See this [SO answer]
(https://stackoverflow.com/questions/75388232/install-python-package-inside-a-jupyter-notebook-kernel) for reference