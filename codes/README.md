# PPG2ABP Codes
The pipeline of PPG2ABP has been implemented in Python v3.5.2, and the deep learning models have been implemented in Keras with Tnesorflow backend.

## Contents

In this README, the pipeline of PPG2ABP is described, which primarily revolves around the following scripts

1. data_processing.py : This script processes and undersamples the data
2. data_handling.py : This script breaks the data into splits to train the deep networks
3. models.py : This script contains the several model definitions
4. metrics.py : Auxiliary script, contains metric computation
5. helper_functions.py : Auxiliary script, contains helper functions
6. train_models.py : This script trains the models using 10 fold cross validaion
7. predict_test.py : This script generates predictions for the test data
8. evaluate.py : This script evaluates PPG2ABP under several metrices


## System Requirements

The experiments have been conducted in a desktop computer with intel core i7-7700 processor (3.6 GHz, 8 MB cache) CPU, 16 GB RAM, and NVIDIA TITAN XP (12 GB, 1582 MHz) GPU. The computer runs on Ubuntu 16.04 Operating System.

A GPU would be essential to train the model, but not necessary when merely testing it.



## Installing Requirements

Before using the pipeline please install the following python modules

> h5py == 2.9.0  
> numpy == 1.17.0  
> tqdm == 4.19.5  
> matplotlib == 2.2.3  
> seaborn == 0.9.0  
> scipy == 1.4.1  
> scikit-learn == 0.19.1  
> tensorflow-gpu == 1.5.0  
> Keras == 2.2.4  
> Keras-Applications == 1.0.8  
> Keras-Preprocessing == 1.1.0  


Install the requirements by using the following command

```
$ pip3 install -r requirements.txt
```


## Downloading the original dataset

The dataset is available in UCI Machine Learning Repository and can be downloaded as follows:


```
$ wget https://archive.ics.uci.edu/ml/machine-learning-databases/00340/data.zip
```

Next, unzip the data. Please make sure you have ***unzip*** installed in your system. The following command will unzip the datafiles into a directory named *raw_data*

```
$ unzip data.zip -d raw_data
```

Alternatively, you may extract the file using some other software and put the extracted files in a folder named *raw_data*


## Smaller Portion of Data to Test PPG2ABP

Please download the downsampled version of the original dataset [data.hdf5](https://drive.google.com/file/d/1IxN2sX2TX0uK6CFDh8eudb8haz3RlF7X/view?usp=sharing) to perform the tests, please refer to [PPG2ABP.ipynb](https://github.com/nibtehaz/PPG2ABP/blob/master/codes/PPG2ABP.ipynb) for further reference

## Expected Results

The Expected results are presented at the last secion of the Jupyter Notebook [PPG2ABP.ipynb](https://github.com/nibtehaz/PPG2ABP/blob/master/codes/PPG2ABP.ipynb), which are identical to that presented in the paper, please refer to [PPG2ABP.ipynb](https://github.com/nibtehaz/PPG2ABP/blob/master/codes/PPG2ABP.ipynb) for further information

## Demo 

You may run PPG2ABP using the comprehensive instructions presented in the [PPG2ABP.ipynb](https://github.com/nibtehaz/PPG2ABP/blob/master/codes/PPG2ABP.ipynb)

## Instructions for Using the Scripts

After downloading the data and unzipping it (putting in the *raw_data* directory), please run the following scripts sequentially

1. Process the data (this may take around 9-11 days on a Intel Core i7-7700 CPU)

```
$ python3 data_processing.py 
```

2. Break the data into splits

```
$ python3 data_handling.py 
```

3. Train the model (this may take around 11-12 days on a NVIDIA TITAN XP GPU)

```
$ python3 train_models.py 
```

4. Predict for the test data

```
$ python3 predict_test.py
```

5. Evaluate PPG2ABP under different metrics

```
$ python3 evaluate.py 
```



## Note

It is a well-known fact that training deep networks using GPUs is a nondeterministic process, therefore training new models on the same data hardly ensures identical configuration of models. Therefore, to achieve the same results as we presented in our paper we request using our provided models and data split (the data splitting were performed in 8 computers parallely to speed up the process, therefore this random process couldn't be seeded properly). Please refer to the Jupyter Notebook for more details.
 
