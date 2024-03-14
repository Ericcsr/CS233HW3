## Getting the Code
All necessary starter code is already made available in this codebase. However, if you would prefer to have a GitHub repo with the codebase (which might make working with Google Cloud easier), we have provided a public repo of the code here which you can fork/clone: [https://github.com/StanfordGeometryLab/cs233_gtda_hw4](https://github.com/StanfordGeometryLab/cs233_gtda_hw4).

## Environment and Code Setup
All requirements are listed in `setup.py`. We will walk through how to install these requirements and set up the code below. The code has been tested with Python 3.6 (but any 3.x version should work), and in these instructions we assume you are using an Ubuntu or MacOS environment (though similar steps can be taken on Windows). Note that some instructions are only relevant if setting up in a GPU environment (as on Google Cloud).

[CPU and GPU] Make sure you are in the top level directory of the codebase (with `setup.py` and this README). First, set up a [virtual environment](https://docs.python.org/3/library/venv.html) that will be used to install all dependencies by running the following commands:
```
sudo apt-get install python3-venv       # install python virtual env on your system
python3 -m venv ./hw4_env               # create an environment called hw4_env
source hw4_env/bin/activate             # activate that environment
pip install --upgrade pip               # upgrade pip
```

[GPU Only] If you are on a GPU system, we will explicitly install PyTorch before anything else to make sure it installs the correct GPU version (using CUDA 11.3):
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

[CPU and GPU] Next, install the codebase as a module:
```
pip install -e .
```

This codebase includes two Chamfer distance implementations, one efficient one that can only run on the GPU and one that is significantly slower but also runs on CPU. When developing your code locally (likely on CPU), you will need to use the slower one while it's recommended to use the faster one when using a GPU.

[CPU Only] To run the starter code using the slower CPU-friendly Chamfer loss, you will need to modify the imports in  `models/pointcloud_autoencoder.py` and `models/part_aware_pointcloud_autoencoder.py` as indicated in those files. Also you will need to use the device `cpu` in your PyTorch code. Make sure to undo all these when moving to the GPU (PyTorch device `cuda`) to use the faster implementation!

[GPU Only] Using the fast Chamfer loss on the GPU requires installing the ninja build system as follows:
```
sudo apt update
sudo apt-get install python3-dev
sudo apt install ninja-build
```

## Getting Started
Start your work at `notebooks/main.ipynb` (or `notebooks_as_python_scripts/main.py` if you are not a fan of notebooks).

If you want to use the notebooks, make sure jupyter is installed with `pip install jupyter` then you can open the notebooks using `jupyter notebook` from the root directory.

Best of luck!

------

Potential Hiccup:

The fast(er) implementation of Chamfer requires the ninja build system installed. You can try `sudo apt install ninja-build` as above, but if you are on an older version of Ubuntu (16.04 or below) you may need to use:
```
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```

If you cannot do it, you might have to resort to the ~10x slower provided implementation of Chamfer in `losses/nn_distance/chamfer_loss` (see notes inside the models/pointcloud_autoencoder.py).

-----