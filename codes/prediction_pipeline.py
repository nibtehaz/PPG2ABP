from helper_functions import *
from metrics import *
from models import *
import time
import pickle
import os

def generate_outputs_val():

    length = 1024

    dt = pickle.load(open(os.path.join('data','val0.p'),'rb'))
    X_val = dt['X_val']
    Y_val = dt['Y_val']    

    dt = pickle.load(open(os.path.join('data','meta0.p'),'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    mdl1 = UNetWide64(length)
    mdl1.load_weights('./models/UNetWide64_model1_25July.h5')

    Y_val_P = mdl1.predict(X_val)
    pickle.dump(Y_val_P,open('val_output_lvl1.p','wb'))

    mdl1 = None
    X_val = None
    Y_val = None

    mdl2 = MultiResUNet1D(length)
    mdl2.load_weights('./models/MultiResUNet1D_model2_25July.h5')

    Y_val_P = mdl2.predict(Y_val_P)
    pickle.dump(Y_val_P,open('val_output_lvl2.p','wb'))

def observe_results_val():

    length = 1024

    dt = pickle.load(open(os.path.join('data','val0.p'),'rb'))
    X_val = dt['X_val']
    Y_val = dt['Y_val']    

    dt = pickle.load(open(os.path.join('data','meta0.p'),'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    Y_val_P = pickle.load(open('./val_output_lvl2.p','rb'))

    while(True):

        observeOutput2(X_val,Y_val,Y_val_P,max_abp, min_abp, max_ppg, min_ppg)

def errorVsSQI_val():

    length = 1024

    dt = pickle.load(open(os.path.join('data','val0.p'),'rb'))
    X_val = dt['X_val']
    Y_val = dt['Y_val']    

    dt = pickle.load(open(os.path.join('data','meta0.p'),'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    Y_val_P = pickle.load(open('./val_output_lvl2.p','rb'))

    

    errorVsSQI2(X_val,Y_val,Y_val_P,max_abp, min_abp, max_ppg, min_ppg)


def generate_outputs_test():

    length = 1024

    dt = pickle.load(open(os.path.join('data','test0.p'),'rb'))
    X_test = dt['X_test']
    Y_test = dt['Y_test']    

    dt = pickle.load(open(os.path.join('data','meta0.p'),'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    mdl1 = UNetWide64(length)
    mdl1.load_weights('./models/UNetWide64_model1_25July.h5')

    Y_test_P = mdl1.predict(X_test)
    pickle.dump(Y_test_P,open('test_output_lvl1.p','wb'))

    mdl1 = None
    X_test = None
    Y_test = None

    mdl2 = MultiResUNet1D(length)
    mdl2.load_weights('./models/MultiResUNet1D_model2_25July.h5')

    Y_test_P = mdl2.predict(Y_test_P)
    pickle.dump(Y_test_P,open('test_output_lvl2.p','wb'))

def observe_results_test():

    length = 1024

    dt = pickle.load(open(os.path.join('data','test0.p'),'rb'))
    X_test = dt['X_test']
    Y_test = dt['Y_test']    

    dt = pickle.load(open(os.path.join('data','meta0.p'),'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    Y_test_P = pickle.load(open('./test_output_lvl2.p','rb'))

    while(True):

        observeOutputDS(X_test,Y_test,Y_test_P,max_abp, min_abp, max_ppg, min_ppg)


def evaluate_performance_test():

    length = 1024

    dt = pickle.load(open(os.path.join('data','test0.p'),'rb'))
    X_test = dt['X_test']
    Y_test = dt['Y_test']    

    dt = pickle.load(open(os.path.join('data','meta0.p'),'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    Y_test_P = pickle.load(open('./test_output_lvl2.p','rb'))

    evaluatePerformanceDS(X_test,Y_test,Y_test_P,max_abp, min_abp, max_ppg, min_ppg)


def main():
    evaluate_performance_test()
    #generate_outputs_test()
    #observe_results_test()
    #errorVsSQI_val()

if __name__ == '__main__':
    main()