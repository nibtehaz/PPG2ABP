"""
    Trains the PPG2ABP model.
    10 fold cross validation scheme is followed.
    Once training is completed the best model based
    on performance on validation data is selected
    and independent test is performed.
"""

from helper_functions import *
from models import *
import time
from tqdm import tqdm
import pickle
import os
from keras.optimizers import Adam

def train_approximate_network():
    """
        Trains the approximate network in 10 fold cross validation manner
    """
    
    model_dict = {}                                             # all the different models
    model_dict['UNet'] = UNet
    model_dict['UNetLite'] = UNetLite
    model_dict['UNetWide40'] = UNetWide40
    model_dict['UNetWide48'] = UNetWide48
    model_dict['UNetDS64'] = UNetDS64
    model_dict['UNetWide64'] = UNetWide64
    model_dict['MultiResUNet1D'] = MultiResUNet1D
    model_dict['MultiResUNetDS'] = MultiResUNetDS


    mdlName1 = 'UNetDS64'                                       # approximation network
    mdlName2 = 'MultiResUNet1D'                                 # refinement network
    
    length = 1024                                               # length of the signal

    try:                                                        # create directory to save models
        os.makedirs('models')
    except:
        pass

    try:                                                        # create directory to save training history
        os.makedirs('History')
    except:
        pass

                                                                    # 10 fold cross validation
    for foldname in range(10):

        print('----------------')
        print('Training Fold {}'.format(foldname+1))
        print('----------------')
                                                                                            # loading training data
        dt = pickle.load(open(os.path.join('data','train{}.p'.format(foldname)),'rb'))
        X_train = dt['X_train']
        Y_train = dt['Y_train']
                                                                                            # loading validation data
        dt = pickle.load(open(os.path.join('data','val{}.p'.format(foldname)),'rb'))
        X_val = dt['X_val']
        Y_val = dt['Y_val']

                                                                                            # loading metadata
        dt = pickle.load(open(os.path.join('data','meta{}.p'.format(foldname)),'rb'))
        max_ppg = dt['max_ppg']
        min_ppg = dt['min_ppg']
        max_abp = dt['max_abp']
        min_abp = dt['min_abp']


        Y_train = prepareLabel(Y_train)                                         # prepare labels for training deep supervision
        
        Y_val = prepareLabel(Y_val)                                             # prepare labels for training deep supervision
    

    
        mdl1 = model_dict[mdlName1](length)             # create approximation network

                                                                            # loss = mae, with deep supervision weights
        mdl1.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mean_squared_error'], loss_weights=[1., 0.9, 0.8, 0.7, 0.6])                                                         


        checkpoint1_ = ModelCheckpoint(os.path.join('models','{}_model1_fold{}.h5'.format(mdlName1,foldname)), verbose=1, monitor='val_out_loss',save_best_only=True, mode='auto')  
                                                                        # train approximation network for 100 epochs
        history1 = mdl1.fit(X_train,{'out': Y_train['out'], 'level1': Y_train['level1'], 'level2':Y_train['level2'], 'level3':Y_train['level3'] , 'level4':Y_train['level4']},epochs=100,batch_size=256,validation_data=(X_val,{'out': Y_val['out'], 'level1': Y_val['level1'], 'level2':Y_val['level2'], 'level3':Y_val['level3'] , 'level4':Y_val['level4']}),callbacks=[checkpoint1_],verbose=1)

        pickle.dump(history1, open('History/{}_model1_fold{}.p'.format(mdlName1,foldname),'wb'))    # save training history


        mdl1 = None                                             # garbage collection

        time.sleep(300)                                         # pause execution for a while to free the gpu
    


def train_refinement_network():
    """
        Trains the refinement network in 10 fold cross validation manner
    """
    
    model_dict = {}                                             # all the different models
    model_dict['UNet'] = UNet
    model_dict['UNetLite'] = UNetLite
    model_dict['UNetWide40'] = UNetWide40
    model_dict['UNetWide48'] = UNetWide48
    model_dict['UNetDS64'] = UNetDS64
    model_dict['UNetWide64'] = UNetWide64
    model_dict['MultiResUNet1D'] = MultiResUNet1D
    model_dict['MultiResUNetDS'] = MultiResUNetDS


    mdlName1 = 'UNetDS64'                                       # approximation network
    mdlName2 = 'MultiResUNet1D'                                 # refinement network
    
    length = 1024                                               # length of the signal

                                                                    # 10 fold cross validation
    for foldname in range(10):

        print('----------------')
        print('Training Fold {}'.format(foldname+1))
        print('----------------')
                                                                                            # loading training data
        dt = pickle.load(open(os.path.join('data','train{}.p'.format(foldname)),'rb'))
        X_train = dt['X_train']
        Y_train = dt['Y_train']
                                                                                            # loading validation data
        dt = pickle.load(open(os.path.join('data','val{}.p'.format(foldname)),'rb'))
        X_val = dt['X_val']
        Y_val = dt['Y_val']

                                                                                            # loading metadata
        dt = pickle.load(open(os.path.join('data','meta{}.p'.format(foldname)),'rb'))
        max_ppg = dt['max_ppg']
        min_ppg = dt['min_ppg']
        max_abp = dt['max_abp']
        min_abp = dt['min_abp']


        Y_train = prepareLabel(Y_train)                                         # prepare labels for training deep supervision
        
        Y_val = prepareLabel(Y_val)                                             # prepare labels for training deep supervision
    
    
        mdl1 = model_dict[mdlName1](length)                 # load approximation network
        mdl1.load_weights(os.path.join('models','{}_model1_fold{}.h5'.format(mdlName1,foldname)))   # load weights

        X_train = prepareDataDS(mdl1, X_train)          # prepare training data for 2nd stage, considering deep supervision
        X_val = prepareDataDS(mdl1, X_val)              # prepare validation data for 2nd stage, considering deep supervision

        mdl1 = None                                 # garbage collection

    
        mdl2 = model_dict[mdlName2](length)            # create refinement network

                                                                    # loss = mse
        mdl2.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'])

        checkpoint2_ = ModelCheckpoint(os.path.join('models','{}_model2_fold{}.h5'.format(mdlName2,foldname)), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  

                                                                # train refinement network for 100 epochs
        history2 = mdl2.fit(X_train,Y_train['out'],epochs=100,batch_size=192,validation_data=(X_val,Y_val['out']),callbacks=[checkpoint2_])

        pickle.dump(history2, open('History/{}_model2_fold{}.p'.format(mdlName2,foldname),'wb'))    # save training history

        time.sleep(300)                                         # pause execution for a while to free the gpu





def main():

    train_approximate_network()             # train the approximate models for 10 fold
    train_refinement_network()             # train the refinement models for 10 fold

if __name__ == '__main__':
    main()