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
    model_dict['UNetWide64'] = UNetWide64
    model_dict['MultiResUNet1D'] = MultiResUNet1D

    date2 = '8Aug_msemse'
    date1 = '31July_maemse'
    date2 = date1

    mdlName2 = 'MultiResUNet1D'
    mdlName1 = 'UNetWide64'
    mdlName3 = 'UNetWide64'
    mdlName4 = 'UNetWide64'
    mdlName1 = mdlName2

    length = 1024

    dt = pickle.load(open(os.path.join('data','train0.p'),'rb'))
    X_train = dt['X_train']
    Y_train = dt['Y_train']

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


    ###################### Stage 1 ##################################
    if(True):  
        mdl1 = model_dict[mdlName1](length)

        #mdl1.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mean_squared_error'])
        mdl1.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'])

        try:
            os.makedirs('models')
        except:
            pass

        checkpoint1_ = ModelCheckpoint(os.path.join('models','{}_model1_{}.h5'.format(mdlName1,date1)), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
        
        history1 = mdl1.fit(X_train,Y_train,epochs=100,batch_size=192,validation_data=(X_val,Y_val),callbacks=[checkpoint1_])

        pickle.dump(history1, open('History/{}_model1_{}.p'.format(mdlName1,date1),'wb'))

        metrics = evaluatePerformance(mdl1, X_val , Y_val, max_abp, min_abp, max_ppg, min_ppg)
        fp = open('Logs/Val_{}_model1_{}.txt'.format(mdlName1,date1),'w')
        fp.write(metrics)
        fp.close()

        metrics = evaluatePerformance(mdl1, X_test , Y_test, max_abp, min_abp, max_ppg, min_ppg)
        fp = open('Logs/Test_{}_model1_{}.txt'.format(mdlName1,date1),'w')
        fp.write(metrics)
        fp.close()

        time.sleep(300)

    ###################### Stage 2 ##################################

    
    mdl1 = model_dict[mdlName1](length)
    mdl1.load_weights(os.path.join('models','{}_model1_{}.h5'.format(mdlName1,date1)))


    (X_train, X_val, X_test) = prepareData(mdl1, X_train, X_val, X_test, Y_train, Y_val, Y_test)

    mdl1 = None
    
    if(True):    
        mdl2 = model_dict[mdlName2](length)

        mdl2.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'])

        checkpoint2_ = ModelCheckpoint(os.path.join('models','{}_model2_{}.h5'.format(mdlName2,date2)), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
        
        history2 = mdl2.fit(X_train,Y_train,epochs=100,batch_size=192,validation_data=(X_val,Y_val),callbacks=[checkpoint2_])

        pickle.dump(history2, open('History/{}_model2_{}.p'.format(mdlName2,date2),'wb'))

    mdl2 = model_dict[mdlName2](length)
    mdl2.load_weights(os.path.join('models','{}_model2_{}.h5'.format(mdlName2,date2)))

    metrics = evaluatePerformance(mdl2, X_val , Y_val, max_abp, min_abp, max_ppg, min_ppg)
    fp = open('Logs/Val_{}_model2_{}.txt'.format(mdlName2,date2),'w')
    fp.write(metrics)
    fp.close()

    metrics = evaluatePerformance(mdl2, X_test , Y_test, max_abp, min_abp, max_ppg, min_ppg)
    fp = open('Logs/Test_{}_model2_{}.txt'.format(mdlName2,date2),'w')
    fp.write(metrics)
    fp.close()

    time.sleep(300)
    '''
    ###################### Stage 3 ##################################

    mdl2 = model_dict[mdlName2](length)
    mdl2.load_weights(os.path.join('models','{}_model2_{}.h5'.format(mdlName2,date)))

    (X_train, X_val, X_test) = prepareData(mdl2, X_train, X_val, X_test, Y_train, Y_val, Y_test)

    mdl3 = model_dict[mdlName3](length)
    mdl2 = None

    

    mdl3.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'])

    checkpoint3_ = ModelCheckpoint(os.path.join('models','{}_model3_{}.h5'.format(mdlName3,date)), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
    
    history3 = mdl3.fit(X_train,Y_train,epochs=100,batch_size=192,validation_data=(X_val,Y_val),callbacks=[checkpoint3_])

    pickle.dump(history3, open('History/{}_model3_{}.p'.format(mdlName3,date),'wb'))

    metrics = evaluatePerformance(mdl3, X_val , Y_val, max_abp, min_abp, max_ppg, min_ppg)
    fp = open('Logs/Val_{}_model3_{}.txt'.format(mdlName3,date),'w')
    fp.write(metrics)
    fp.close()

    metrics = evaluatePerformance(mdl3, X_test , Y_test, max_abp, min_abp, max_ppg, min_ppg)
    fp = open('Logs/Test_{}_model3_{}.txt'.format(mdlName3,date),'w')
    fp.write(metrics)
    fp.close()

    time.sleep(300)


    ###################### Stage 4 ##################################

    mdl3 = model_dict[mdlName3](length)
    mdl3.load_weights(os.path.join('models','{}_model3_{}.h5'.format(mdlName1,date)))

    (X_train, X_val, X_test) = prepareData(mdl3, X_train, X_val, X_test, Y_train, Y_val, Y_test)

    mdl4 = model_dict[mdlName4](length)
    mdl3 = None
    

    mdl4.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'])

    checkpoint4_ = ModelCheckpoint(os.path.join('models','{}_model4_{}.h5'.format(mdlName4,date)), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
    
    history4 = mdl4.fit(X_train,Y_train,epochs=100,batch_size=192,validation_data=(X_val,Y_val),callbacks=[checkpoint4_])

    pickle.dump(history4, open('History/{}_model4_{}.p'.format(mdlName4,date),'wb'))

    metrics = evaluatePerformance(mdl4, X_val , Y_val, max_abp, min_abp, max_ppg, min_ppg)
    fp = open('Logs/Val_{}_model4_{}.txt'.format(mdlName4,date),'w')
    fp.write(metrics)
    fp.close()

    metrics = evaluatePerformance(mdl4, X_test , Y_test, max_abp, min_abp, max_ppg, min_ppg)
    fp = open('Logs/Test_{}_model4_{}.txt'.format(mdlName4,date),'w')
    fp.write(metrics)
    fp.close()

    time.sleep(300)

    '''

if __name__ == '__main__':
    main()