# Make and/or load ResNet-15 embeddings of CIFAR-10 and SVHN

## Set Up on Linux:

1. Download conda at https://docs.conda.io/en/latest/
2. run `sh setup.sh` on a linux system, preferably Ubuntu
3. run `source activate py36` to enter the conda env
4. to get an interactive shell with dicts containing embeddings for each dataset, run `python -i load_embeddings.py`.

## Set Up on MacOS:

1. Download conda at https://docs.conda.io/en/latest/
2. run `conda create -n py36 python=3.6`
3. run `source activate py36` to enter the conda env
4. `pip install -r requirements.txt`
5. to get an interactive shell with dicts containing embeddings for each dataset, run `python -i load_embeddings.py`.
