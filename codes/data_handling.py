"""
    Prepares the data splits for 10 fold cross validation
"""

import h5py
import numpy as np
import os
from tqdm import tqdm
import pickle


def fold_data():
    """
        folds the data into splits and saves them
        to perform 10 fold cross validation
    """

    length = 1024           # length of the signals

                                # we take this starting points of validation data
                                # as we have already shuffled the episodes while creating
                                # the data.hdf5 file
    validation_data_start = {
        0: 90000,
        1: 0,
        2: 10000,
        3: 20000,
        4: 30000,
        5: 40000,
        6: 50000,
        7: 60000,
        8: 70000,
        9: 80000,
    }

    for fold_id in tqdm(range(10), desc='Folding Data'):        # iterate for 10 folds

        fl = h5py.File(os.path.join('data', 'data.hdf5'), 'r')      # load the episode data

        X_train = []                        # intialize train data
        Y_train = []

        X_val = []                          # intialize validation data
        Y_val = []

        max_ppg = -10000                    # intialize metadata, min-max of abp,ppg signals
        min_ppg = 10000
        max_abp = -10000
        min_abp = 10000

        val_start = validation_data_start[fold_id]      # validation data start
        val_end = val_start + 10000                     # validation data end

        for i in tqdm(range(0, val_start), desc='Training Data Part 1'):    # training samples before validation samples

            X_train.append(np.array(fl['data'][i][1][:length]).reshape(length, 1))  # ppg signal
            Y_train.append(np.array(fl['data'][i][0][:length]).reshape(length, 1))  # abp signal

            max_ppg = max(max(fl['data'][i][1]), max_ppg)       # update min-max of ppg 
            min_ppg = min(min(fl['data'][i][1]), min_ppg)

            max_abp = max(max(fl['data'][i][0]), max_abp)       # update min-max of abp
            min_abp = min(min(fl['data'][i][0]), min_abp)

        
        for i in tqdm(range(val_end, 100000), desc='Training Data Part 2'):    # training samples after validation samples

            X_train.append(np.array(fl['data'][i][1][:length]).reshape(length, 1))  # ppg signal
            Y_train.append(np.array(fl['data'][i][0][:length]).reshape(length, 1))  # abp signal

            max_ppg = max(max(fl['data'][i][1]), max_ppg)       # update min-max of ppg 
            min_ppg = min(min(fl['data'][i][1]), min_ppg)

            max_abp = max(max(fl['data'][i][0]), max_abp)       # update min-max of abp
            min_abp = min(min(fl['data'][i][0]), min_abp)

        
        for i in tqdm(range(val_start, val_end), desc='Validation Data'):

            X_val.append(np.array(fl['data'][i][1][:length]).reshape(length, 1))  # ppg signal
            Y_val.append(np.array(fl['data'][i][0][:length]).reshape(length, 1))  # abp signal

            max_ppg = max(max(fl['data'][i][1]), max_ppg)       # update min-max of ppg 
            min_ppg = min(min(fl['data'][i][1]), min_ppg)

            max_abp = max(max(fl['data'][i][0]), max_abp)       # update min-max of abp
            min_abp = min(min(fl['data'][i][0]), min_abp)


        fl = None                   # garbage collection


        X_train = np.array(X_train)             # converting to numpy array
        X_train -= min_ppg                      # normalizing
        X_train /= (max_ppg-min_ppg)

        Y_train = np.array(Y_train)             # converting to numpy array
        Y_train -= min_abp                      # normalizing
        Y_train /= (max_abp-min_abp)

                                                                # saving the training data split
        pickle.dump({'X_train': X_train, 'Y_train': Y_train}, open(os.path.join('data', 'train{}.p'.format(fold_id)), 'wb'))

        X_train = []                   # garbage collection
        Y_train = []

        X_val = np.array(X_val)                 # converting to numpy array        
        X_val -= min_ppg                        # normalizing
        X_val /= (max_ppg-min_ppg)

        Y_val = np.array(Y_val)                 # converting to numpy array
        Y_val -= min_abp                        # normalizing
        Y_val /= (max_abp-min_abp)

                                                                # saving the validation data split
        pickle.dump({'X_val': X_val, 'Y_val': Y_val}, open(os.path.join('data', 'val{}.p'.format(fold_id)), 'wb'))

        X_val = []                   # garbage collection
        Y_val = []
                                                                # saving the metadata
        pickle.dump({'max_ppg': max_ppg,
                     'min_ppg': min_ppg,
                     'max_abp': max_abp,
                     'min_abp': min_abp}, open(os.path.join('data', 'meta{}.p'.format(fold_id)), 'wb'))

    fl = h5py.File(os.path.join('data', 'data.hdf5'), 'r')      # loading the episode data

    X_test = []                 # intialize test data
    Y_test = []

    for i in tqdm(range(100000, len(fl['data']))):

        X_test.append(np.array(fl['data'][i][1][:length]).reshape(length, 1))       # ppg signal
        Y_test.append(np.array(fl['data'][i][0][:length]).reshape(length, 1))       # abp signal

        max_ppg = max(max(fl['data'][i][1]), max_ppg)
        min_ppg = min(min(fl['data'][i][1]), min_ppg)

        max_abp = max(max(fl['data'][i][0]), max_abp)
        min_abp = min(min(fl['data'][i][0]), min_abp)

    X_test = np.array(X_test)           # converting to numpy array
    X_test -= min_ppg                   # normalizing
    X_test /= (max_ppg-min_ppg)
    
    Y_test = np.array(Y_test)           # converting to numpy array
    Y_test -= min_abp                   # normalizing
    Y_test /= (max_abp-min_abp)

                                                                # saving the test data split
    pickle.dump({'X_test': X_test,'Y_test': Y_test}, open(os.path.join('data', 'test.p'), 'wb'))


def main():
    
    fold_data()         # splits the data for 10 fold cross validation


if __name__ == '__main__':
    main()
