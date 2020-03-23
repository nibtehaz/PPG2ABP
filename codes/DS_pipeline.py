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

    #dt = pickle.load(open(os.path.join('data','train0.p'),'rb'))
    #X_train = dt['X_train']
    #Y_train = dt['Y_train']

    dt = pickle.load(open(os.path.join('data','val0.p'),'rb'))
    X_val = dt['X_val']
    Y_val = dt['Y_val']

    dt = pickle.load(open(os.path.join('data','test0.p'),'rb'))
    X_test = dt['X_test']
    Y_test = dt['Y_test']

    dt = pickle.load(open(os.path.join('data','meta0.p'),'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    #Y_train = prepareLabel(Y_train)
    #print('____')
    #Y_val = prepareLabel(Y_val)
    #print('____')
    #Y_test = prepareLabel(Y_test)
    #print('____')

    #pickle.dump(Y_train,open(os.path.join('data','ytrain0.p'),'wb'))
    #pickle.dump(Y_val,open(os.path.join('data','yval0.p'),'wb'))
    #pickle.dump(Y_test,open(os.path.join('data','ytest0.p'),'wb'))

    #Y_train = pickle.load(open(os.path.join('data','ytrain0.p'),'rb'))
    Y_val = pickle.load(open(os.path.join('data','yval0.p'),'rb'))
    Y_test = pickle.load(open(os.path.join('data','ytest0.p'),'rb'))


    ###################### Stage 1 ##################################
    if(False):  
        mdl1 = model_dict[mdlName1](length)

        #mdl1.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mean_squared_error'])
        
        mdl1.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mean_squared_error'], loss_weights=[1., 0.9, 0.8, 0.7, 0.6])

        try:
            os.makedirs('models')
        except:
            pass

        checkpoint1_ = ModelCheckpoint(os.path.join('models','{}_model1_{}.h5'.format(mdlName1,date1)), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
        
        history1 = mdl1.fit(X_train,{'out': Y_train['out'], 'level1': Y_train['level1'], 'level2':Y_train['level2'], 'level3':Y_train['level3'] , 'level4':Y_train['level4']},epochs=100,batch_size=256,validation_data=(X_val,{'out': Y_val['out'], 'level1': Y_val['level1'], 'level2':Y_val['level2'], 'level3':Y_val['level3'] , 'level4':Y_val['level4']}),callbacks=[checkpoint1_],verbose=1)

        pickle.dump(history1, open('History/{}_model1_{}.p'.format(mdlName1,date1),'wb'))

    mdl1 = model_dict[mdlName1](length)
    mdl1.load_weights(os.path.join('models','{}_model1_{}.h5'.format(mdlName1,date1)))    


    if(True):

        #metrics = evaluatePerformanceDS(mdl1, X_val , Y_val['out'], max_abp, min_abp, max_ppg, min_ppg)
        #fp = open('Logs/Val_{}_model1_{}.txt'.format(mdlName1,date1),'w')
        #fp.write(metrics)
        #fp.close()
        
        metrics = evaluatePerformanceDS(mdl1, X_test , Y_test['out'], max_abp, min_abp, max_ppg, min_ppg)
        fp = open('Logs/Test_{}_model1_{}.txt'.format(mdlName1,date1),'w')
        fp.write(metrics)
        fp.close()        

    ###################### Stage 2 ##################################


    ([], X_val, X_test) = prepareDataDS(mdl1, [], X_val, X_test, [], Y_val['out'], Y_test['out'])

    mdl1 = None
    
    if(False):    
        mdl2 = model_dict[mdlName2](length)

        mdl2.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'], loss_weights=[1., 0.9, 0.8, 0.7, 0.6])

        checkpoint2_ = ModelCheckpoint(os.path.join('models','{}_model2_{}.h5'.format(mdlName2,date2)), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
        
        history2 = mdl2.fit(X_train,{'out': Y_train['out'], 'level1': Y_train['level1'], 'level2':Y_train['level2'], 'level3':Y_train['level3'] , 'level4':Y_train['level4']},epochs=100,batch_size=200,validation_data=(X_val,{'out': Y_val['out'], 'level1': Y_val['level1'], 'level2':Y_val['level2'], 'level3':Y_val['level3'] , 'level4':Y_val['level4']}),callbacks=[checkpoint2_],verbose=1)

        pickle.dump(history2, open('History/{}_model2_{}.p'.format(mdlName2,date2),'wb'))

    mdl2 = model_dict[mdlName2](length)
    mdl2.load_weights(os.path.join('models','{}_model2_{}.h5'.format(mdlName2,date2)))

    if(True):

        #metrics = evaluatePerformanceDS(mdl2, X_val , Y_val['out'], max_abp, min_abp, max_ppg, min_ppg)
        #fp = open('Logs/Val_{}_model2_{}.txt'.format(mdlName2,date2),'w')
        #fp.write(metrics)
        #fp.close()

        metrics = evaluatePerformanceDS(mdl2, X_test , Y_test['out'], max_abp, min_abp, max_ppg, min_ppg)
        fp = open('Logs/Test_{}_model2_{}.txt'.format(mdlName2,date2),'w')
        fp.write(metrics)
        fp.close()

   

if __name__ == '__main__':
    main()