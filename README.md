Auditory-eeg-challenge-2024-code
================================
This is the codebase for the [2024 ICASSP Auditory EEG challenge](https://exporl.github.io/auditory-eeg-challenge-2024).
This codebase contains baseline models and code to preprocess stimuli for both tasks.

# Prerequisites

Python >= 3.6

# General setup

Steps to get a working setup:

## 1. Clone this repository and install the [requirements.txt](requirements.txt)
```bash
git clone https://github.com/exporl/LSTM-model-for-ML-project
cd LSTM-model-for-ML-project
conda create --name tf-LSTM-model python=3.8 
conda activatae tf-LSTM-model
pip install -r requirements.txt
```

## 2. [Download the data](https://homes.esat.kuleuven.be/~lbollens/)

You will need a password, which you will receive when you [register](https://exporl.github.io/auditory-eeg-challenge-2024/registration/).
The folder contains multiple folders (and `zip` files containing the same data as their corresponding folders). For bulk downloading, we recommend using the `zip` files, 

   1. `split_data(.zip)` contains already preprocessed, split and normalized data; ready for model training/evaluation. 
If you want to get started quickly, you can opt to only download this folder/zipfile.

   2. `preprocessed_eeg(.zip)` and `preprocessed_stimuli(.zip)` contain preprocessed EEG and stimuli files (envelope and mel features) respectively.
At this stage data is not yet split into different sets and normalized. To go from this to the data in `split_data`, you will have to run the `split_and_normalize.py` script ([preprocessing_code/split_and_normalize.py](./preprocessing_code/split_and_normalize.py) )

   3. `sub_*(.zip)` and `stimuli(.zip)` contain the raw EEG and stimuli files. 
If you want to recreate the preprocessing steps, you will need to download these files and then run `sparrKULee.py` [(preprocessing_code/sparrKULee.py)](./preprocessing_code/sparrKULee.py) to preprocess the EEG and stimuli and then run the `split_and_normalize.py` script to split and normalize the data.
It is possible to adapt the preprocessing steps in `sparrKULee.py` to your own needs, by adding/removing preprocessing steps. For more detailed information on the pipeline, see the [brain_pipe documentation](https://exporl.github.io/brain_pipe/).


Note that it is possible to use the same preprocessed (and split) dataset for both task 1 and task 2, but it is not required.



## 3. Adjust the `config.json` accordingly

There is a general `config.json` defining the folder names and structure for the data (i.e. [util/config.json](./util/config.json) ).
Adjust `dataset_folder` in the `config.json` file from `null` to the absolute path to the folder containing all data (The `challenge_folder` from the previous point). 
If you follow the BIDS structure, by downloading the whole dataset, the folders preprocessed_eeg, preprocessed_stimuli and split_data, should be located inside the 'derivatives' folder. If you only download these three folders, make sure they are either in a subfolder 'derivatives', or change the 'derivatives' folder in the config, otherwise you will get a file-not-found error when trying to run the experiments. 
  

OK, you should be all setup now!

## 4. Train the model 
```bash
python task1_match_mismatch\experiments\dilated_convolutional_model.py
```

## 5. Generate testset
```bash
python task1_match_mismatch\experiments\test_match_mismatch.py
```
