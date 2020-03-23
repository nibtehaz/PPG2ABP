from helper_functions import *
from metrics import *
from models import *
import time
import pickle
import os
from keras.optimizers import Adam

def main():
    
    model_dict = {}
    model_dict['UNet'] = UNet
    model_dict['UNetLite'] = UNetLite
    model_dict['UNetWide40'] = UNetWide40
    model_dict['UNetWide48'] = UNetWide48
    model_dict['UNetDS64'] = UNetDS64
    model_dict['UNetWide64'] = UNetWide64
    model_dict['MultiResUNet1D'] = MultiResUNet1D
    model_dict['MultiResUNetDS'] = MultiResUNetDS

    date1 = '10Aug_msemseDS'
    date2 = '10Aug_msemseDS'

    mdlName1 = 'UNetDS64'
    mdlName2 = 'MultiResUNetDS'
    
    

    length = 1024

    dt = pickle.load(open(os.path.join('data','val0.p'),'rb'))
    X_val = dt['X_val']
    Y_val = dt['Y_val']

    #dt = pickle.load(open(os.path.join('data','test0.p'),'rb'))
    #X_test = dt['X_test']
    #Y_test = dt['Y_test']

    dt = pickle.load(open(os.path.join('data','meta0.p'),'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

   
    Y_val = pickle.load(open(os.path.join('data','yval0.p'),'rb'))
    #Y_test = pickle.load(open(os.path.join('data','ytest0.p'),'rb'))


    mdl1 = model_dict[mdlName1](length)
    mdl1.load_weights(os.path.join('models','{}_model1_{}.h5'.format(mdlName1,date1)))    

    

    mdl2 = model_dict[mdlName2](length)
    mdl2.load_weights(os.path.join('models','{}_model2_{}.h5'.format(mdlName2,date2)))

    
    



    while True:

        index = int(input('Index='))

        X = np.array([X_val[index]])
        Y = np.array([Y_val['out'][index]])

        observeOutputDS(mdl1, mdl2, X, Y, max_abp, min_abp, max_ppg, min_ppg)

   

if __name__ == '__main__':
    main()