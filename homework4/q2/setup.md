Please execute the commands below in the terminal, one line at a time.

```bash

conda create -n a4q2 python=3.9 -y
conda activate a4q2
conda update -n base -c conda-forge conda # If you encounter an error here, it's okay to skip this command.
conda install pytorch torchvision torchaudio -c pytorch -y
pip install tqdm requests importlib-metadata filelock scikit-learn tokenizers numpy

```