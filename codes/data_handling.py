import h5py
import numpy as np
import os
from tqdm import tqdm
import pickle

def load_raw_data(fl_path):

    fl = h5py.File(fl_path, 'r')

    return fl


def fold_data(fl):

    length = 1024

    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    X_test = []
    Y_test = []

    max_ppg = -10000
    min_ppg = 10000
    max_abp = -10000
    min_abp = 10000


    training_end = 90000
    validation_end = 100000
        

    for i in tqdm(range(training_end)):
        
        X_train.append(np.array(fl['data'][i][1][:length]).reshape(length,1))
        Y_train.append(np.array(fl['data'][i][0][:length]).reshape(length,1))
        
        max_ppg = max(max(fl['data'][i][1]),max_ppg)
        min_ppg = min(min(fl['data'][i][1]),min_ppg)
        
        max_abp = max(max(fl['data'][i][0]),max_abp)
        min_abp = min(min(fl['data'][i][0]),min_abp)
        
    for i in tqdm(range(training_end,validation_end)):
        
        X_val.append(np.array(fl['data'][i][1][:length]).reshape(length,1))
        Y_val.append(np.array(fl['data'][i][0][:length]).reshape(length,1))
        
        max_ppg = max(max(fl['data'][i][1]),max_ppg)
        min_ppg = min(min(fl['data'][i][1]),min_ppg)
        
        max_abp = max(max(fl['data'][i][0]),max_abp)
        min_abp = min(min(fl['data'][i][0]),min_abp)

    for i in tqdm(range(validation_end,len(fl['data']))):
        
        X_test.append(np.array(fl['data'][i][1][:length]).reshape(length,1))
        Y_test.append(np.array(fl['data'][i][0][:length]).reshape(length,1))
        
        max_ppg = max(max(fl['data'][i][1]),max_ppg)
        min_ppg = min(min(fl['data'][i][1]),min_ppg)
        
        max_abp = max(max(fl['data'][i][0]),max_abp)
        min_abp = min(min(fl['data'][i][0]),min_abp)
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X_val = np.array(X_val)
    Y_val = np.array(Y_val)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    X_train -= min_ppg
    X_train /= (max_ppg-min_ppg)
    X_val -= min_ppg
    X_val /= (max_ppg-min_ppg)
    X_test -= min_ppg
    X_test /= (max_ppg-min_ppg)

    Y_train -= min_abp
    Y_train /= (max_abp-min_abp)
    Y_val -= min_abp
    Y_val /= (max_abp-min_abp)
    Y_test -= min_abp
    Y_test /= (max_abp-min_abp)


    fl = None   

    pickle.dump({'X_train':X_train,
                 'Y_train':Y_train},open(os.path.join('data','train0.p'),'wb'))

    pickle.dump({'X_val':X_val,
                 'Y_val':Y_val},open(os.path.join('data','val0.p'),'wb'))

    pickle.dump({'X_test':X_test,
                 'Y_test':Y_test},open(os.path.join('data','test0.p'),'wb'))

    pickle.dump({'max_ppg':max_ppg,
                'min_ppg':min_ppg,
                'max_abp':max_abp,
                'min_abp':min_abp},open(os.path.join('data','meta0.p'),'wb'))



def main():
    
    fl = load_raw_data(os.path.join('data','data_big_127260.hdf5'))
    fold_data(fl)

if __name__ == '__main__':
    main()