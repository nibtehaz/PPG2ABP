from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from metrics import *

length = 1024

def prepareData(mdl, X_train, X_val, X_test, Y_train, Y_val, Y_test):
    
    X2_train = []

    X2_val = []

    X2_test = []


    YPs = mdl.predict(X_train)

    for i in tqdm(range(len(X_train))):

        X2_train.append(np.array(YPs[i]))



    YPs = mdl.predict(X_val)

    for i in tqdm(range(len(X_val))):

        X2_val.append(np.array(YPs[i]))



    YPs = mdl.predict(X_test)

    for i in tqdm(range(len(X_test))):

        X2_test.append(np.array(YPs[i]))


    X2_train = np.array(X2_train)

    X2_val = np.array(X2_val)

    X2_test = np.array(X2_test)

    return (X2_train, X2_val, X2_test)


def prepareDataDS(mdl, X_train, X_val, X_test, Y_train, Y_val, Y_test):
    
    X2_train = []

    X2_val = []

    X2_test = []


    #YPs = mdl.predict(X_train)
    
    #for i in tqdm(range(len(X_train))):
        
    #   X2_train.append(np.array(YPs[0][i]))



    YPs = mdl.predict(X_val)

    for i in tqdm(range(len(X_val))):

        X2_val.append(np.array(YPs[0][i]))



    YPs = mdl.predict(X_test)

    for i in tqdm(range(len(X_test))):

        X2_test.append(np.array(YPs[0][i]))


    X2_train = np.array(X2_train)

    X2_val = np.array(X2_val)

    X2_test = np.array(X2_test)

    return (X2_train, X2_val, X2_test)


def prepareLabel(Y):
    
    def approximate(inp,w_len):
        
        op = []
        
        for i in range(0,len(inp),w_len):
        
            op.append(np.mean(inp[i:i+w_len]))
            
        return np.array(op)
    
    out = {}
    out['out'] = []
    out['level1'] = []
    out['level2'] = []
    out['level3'] = []
    out['level4'] = []
    
    
    for y in tqdm(Y):
        
        #cA1, cD1 = pywt.dwt(np.array(y).reshape(length), 'db1')
        
        #cA2, cD2 = pywt.dwt(np.array(cA1), 'db1')
        
        #cA3, cD3 = pywt.dwt(np.array(cA2), 'db1')
        
        #cA4, cD4 = pywt.dwt(np.array(cA3), 'db1')
        
        cA1 = approximate(np.array(y).reshape(length), 2)
        
        cA2 = approximate(np.array(y).reshape(length), 4)
        
        cA3 = approximate(np.array(y).reshape(length), 8)
        
        cA4 = approximate(np.array(y).reshape(length), 16)
        
        
        
        
        out['out'].append(np.array(y.reshape(length,1)))
        out['level1'].append(np.array(cA1.reshape(length//2,1)))
        out['level2'].append(np.array(cA2.reshape(length//4,1)))
        out['level3'].append(np.array(cA3.reshape(length//8,1)))
        out['level4'].append(np.array(cA4.reshape(length//16,1)))
        
    out['out'] = np.array(out['out'])
    out['level1'] = np.array(out['level1'])
    out['level2'] = np.array(out['level2'])
    out['level3'] = np.array(out['level3'])
    out['level4'] = np.array(out['level4'])
    
    
    return out

def observeOutput(data, mdl, X_train, X_val, X_test, Y_train, Y_val, Y_test, max_abp, min_abp, max_ppg, min_ppg):

    index = int(input('Index = ')) # 310

    selected_data = data

    #y_t = Y_train[index].reshape(length)
    #y_p = mdl.predict(np.array([X_train[index]]))[0].reshape(length)

    if selected_data == 'train' :
        y_t = Y_train[index].reshape(length)
        y_p = mdl.predict(np.array([X_train[index]]))[0].reshape(length)

    if selected_data == 'val' :
        y_t = Y_val[index].reshape(length)
        y_p = mdl.predict(np.array([X_val[index]]))[0].reshape(length)

    if selected_data == 'test' :
        y_t = Y_test[index].reshape(length)
        y_p = mdl.predict(np.array([X_test[index]]))[0].reshape(length)

    plt.figure(figsize=(18,5))
    plt.plot(y_t*max_abp+min_abp,c='b',label='GT')
    plt.plot(y_p*max_abp+min_abp,c='g',label='Output')
    plt.legend()

    plt.figure(figsize=(18,5))
    plt.plot(X_train[index])

    plt.figure(figsize=(18,5))
    plt.subplot(1,2,1)
    plt.plot(y_t)
    plt.subplot(1,2,2)
    plt.plot(y_p)


    print('SBP : True = {} \t Predicted = {} \t Error = {}'.format(round(min(y_t)*max_abp+min_abp,2),round(min(y_p)*max_abp+min_abp,2),round(abs(min(y_t)-min(y_p))*max_abp,2)))
    print('DBP : True = {} \t Predicted = {} \t Error = {}'.format(round(max(y_t)*max_abp+min_abp,2),round(max(y_p)*max_abp+min_abp,2),round(abs(max(y_t)-max(y_p))*max_abp,2)))
    print('MAE : {}'.format(round(np.mean(np.abs(y_t-y_p))*max_abp,2)))
    
def observeOutputDS(mdl1, mdl2, X, Y, max_abp, min_abp, max_ppg, min_ppg):

    yp1 = mdl1.predict(X)

    yp2 = mdl2.predict(np.array([yp1[0][0]]))


    

    plt.figure(figsize=(18,5))
    plt.subplot(2,1,1)    
    plt.plot(Y[0].reshape(1024)*max_abp+min_abp,c='r',label='GT')
    plt.plot(yp1[0][0].reshape(1024)*max_abp+min_abp,c='g',label='Level 1')
    plt.plot(yp2[0][0].reshape(1024)*max_abp+min_abp,c='b',label='Level 2')    
    plt.legend()
    plt.subplot(2,1,2)    
    plt.plot(X[0].reshape(1024)*max_ppg+min_ppg,c='k',label='Input PPG')

    y_t = Y[0].reshape(1024)
    y_p = yp1[0][0].reshape(1024)

    print('-'*25)

    print('SBP : True = {} \t Predicted = {} \t Error = {}'.format(round(min(y_t)*max_abp+min_abp,2),round(min(y_p)*max_abp+min_abp,2),round(abs(min(y_t)-min(y_p))*max_abp,2)))
    print('DBP : True = {} \t Predicted = {} \t Error = {}'.format(round(max(y_t)*max_abp+min_abp,2),round(max(y_p)*max_abp+min_abp,2),round(abs(max(y_t)-max(y_p))*max_abp,2)))
    print('MAE : {}'.format(round(np.mean(np.abs(y_t-y_p))*max_abp,2)))

    y_p = yp2[0][0].reshape(1024)

    print('.'*25)

    print('SBP : True = {} \t Predicted = {} \t Error = {}'.format(round(min(y_t)*max_abp+min_abp,2),round(min(y_p)*max_abp+min_abp,2),round(abs(min(y_t)-min(y_p))*max_abp,2)))
    print('DBP : True = {} \t Predicted = {} \t Error = {}'.format(round(max(y_t)*max_abp+min_abp,2),round(max(y_p)*max_abp+min_abp,2),round(abs(max(y_t)-max(y_p))*max_abp,2)))
    print('MAE : {}'.format(round(np.mean(np.abs(y_t-y_p))*max_abp,2)))
    
    print('-'*25)

    plt.show()

def evaluatePerformance(mdl, X, Y, max_abp, min_abp, max_ppg, min_ppg):

    sbps = []
    dbps = []
    maes = []
    gt = []

    hist = []

    for i in tqdm(range(len(Y))):
        y_t = Y[i].reshape(length)
        y_p = mdl.predict(np.array([X[i]]))[0].reshape(length)

        dbps.append(max_abp*abs(min(y_t)-min(y_p)))
        sbps.append(max_abp*abs(max(y_t)-max(y_p)))
        maes.append(np.mean(np.abs(y_t-y_p))*max_abp)
        gt.append(max_abp*min(y_t)+min_abp)
        hist.append(max_abp*(max(y_t)-max(y_p)))


    print('SBP stat : ',np.mean(sbps),np.std(sbps))
    print('BHS', BHS_standard(sbps))
    print('DBP stat : ',np.mean(dbps),np.std(dbps))
    print('BHS', BHS_standard(dbps))
    print('MAE ',np.mean(maes),np.std(maes))
    
    return 'SBP stat : {} +- {}'.format(np.mean(sbps),np.std(sbps)) + '\n' + 'BHS : {}'.format(BHS_standard(sbps)) + '\n' + 'DBP stat : {} +- {}'.format(np.mean(dbps),np.std(dbps)) + '\n' + 'BHS : {}'.format(BHS_standard(dbps)) + '\n' + 'MAE : {} +- {}'.format(np.mean(maes),np.std(maes))


def evaluatePerformanceDS(mdl, X, Y, max_abp, min_abp, max_ppg, min_ppg):


    sbps = []
    dbps = []
    maes = []
    gt = []

    hist = []

    for i in tqdm(range(len(Y))):
        y_t = Y[i].reshape(length)
        y_p = mdl.predict(np.array([X[i]]))[0][0].reshape(length)

        dbps.append(max_abp*abs(min(y_t)-min(y_p)))
        sbps.append(max_abp*abs(max(y_t)-max(y_p)))
        maes.append(np.mean(np.abs(y_t-y_p))*max_abp)
        gt.append(max_abp*min(y_t)+min_abp)
        hist.append(max_abp*(max(y_t)-max(y_p)))


    print('SBP stat : ',np.mean(sbps),np.std(sbps))
    print('BHS', BHS_standard(sbps))
    print('DBP stat : ',np.mean(dbps),np.std(dbps))
    print('BHS', BHS_standard(dbps))
    print('MAE ',np.mean(maes),np.std(maes))
    
    return 'SBP stat : {} +- {}'.format(np.mean(sbps),np.std(sbps)) + '\n' + 'BHS : {}'.format(BHS_standard(sbps)) + '\n' + 'DBP stat : {} +- {}'.format(np.mean(dbps),np.std(dbps)) + '\n' + 'BHS : {}'.format(BHS_standard(dbps)) + '\n' + 'MAE : {} +- {}'.format(np.mean(maes),np.std(maes))

    

    
def errorVsSQI(mdl, X, Y, signals,  max_abp, min_abp, max_ppg, min_ppg):

    sbps = []
    dbps = []
    maes = []
    sqi = []
    gt = []

    hist = []

    for i in tqdm(range(len(Y))):
        y_t = Y[i].reshape(length)
        y_p = mdl.predict(np.array([X[i]]))[0].reshape(length)
        
        sqi.append(skewness(signals[i].reshape(length)))
        dbps.append(max_abp*abs(min(y_t)-min(y_p)))
        sbps.append(max_abp*abs(max(y_t)-max(y_p)))
        

    
    sns.jointplot(sbps, sqi, kind="kde", space=0, color="r")
    plt.xlabel('SBP Error')
    plt.ylabel('SQI')
    plt.figure()
    plt.scatter(sqi, sbps, c="r",alpha="0.3")
    plt.xlabel('SQI')
    plt.ylabel('SBP Error')
    plt.title('SQI vs SBP Error')
    
    
    plt.figure()
    sns.jointplot(dbps, sqi, kind="kde", space=0, color="b")
    plt.xlabel('DBP')
    plt.ylabel('SQI')
    plt.figure()
    plt.scatter(sqi, dbps, c="b",alpha="0.3")
    plt.xlabel('SQI')
    plt.ylabel('DBP Error')
    plt.title('SQI vs DBP Error')
    
    plt.show()



def evaluatePerformanceInd(mdl, X, Y, max_abp, min_abp, max_ppg, min_ppg):

	sbps = []
	dbps = []
	maes = []
	gt = []

	hist = []

	predicted = []

	try:
		predicted = pickle.load(open('Independent_predictions.p','rb'))

	except:
		
		predicted = mdl.predict(X,verbose=2)		

		pickle.dump(predicted,open('Independent_predictions.p','wb'))

	for i in tqdm(range(len(Y))):

		y_t = Y[i].reshape(length)
		y_p = predicted[i].reshape(length)

		dbps.append(max_abp*abs(min(y_t)-min(y_p)))
		sbps.append(max_abp*abs(max(y_t)-max(y_p)))
		maes.append(np.mean(np.abs(y_t-y_p))*max_abp)
		gt.append(max_abp*min(y_t)+min_abp)
		hist.append(max_abp*(max(y_t)-max(y_p)))


	print('SBP stat : ',np.mean(sbps),np.std(sbps))
	print('BHS', BHS_standard(sbps))
	print('DBP stat : ',np.mean(dbps),np.std(dbps))
	print('BHS', BHS_standard(dbps))
	print('MAE ',np.mean(maes),np.std(maes))

	
	
	return 'SBP stat : {} +- {}'.format(np.mean(sbps),np.std(sbps)) + '\n' + 'BHS : {}'.format(BHS_standard(sbps)) + '\n' + 'DBP stat : {} +- {}'.format(np.mean(dbps),np.std(dbps)) + '\n' + 'BHS : {}'.format(BHS_standard(dbps)) + '\n' + 'MAE : {} +- {}'.format(np.mean(maes),np.std(maes))



def evaluatePerformanceTrimInd(mdl, X, Y, max_abp, min_abp, max_ppg, min_ppg):

	sbps = []
	dbps = []
	maes = []
	gt = []

	hist = []

	predicted = []

	try:
		predicted = pickle.load(open('Independent_predictions.p','rb'))

	except:
		
		predicted = mdl.predict(X,verbose=2)		

		pickle.dump(predicted,open('Independent_predictions.p','wb'))

	for i in tqdm(range(len(Y))):

		y_t = Y[i].reshape(length)[256:-256]
		y_p = predicted[i].reshape(length)[256:-256]

		dbps.append(max_abp*abs(min(y_t)-min(y_p)))
		sbps.append(max_abp*abs(max(y_t)-max(y_p)))
		maes.append(np.mean(np.abs(y_t-y_p))*max_abp)
		gt.append(max_abp*min(y_t)+min_abp)
		hist.append(max_abp*(max(y_t)-max(y_p)))


	print('SBP stat : ',np.mean(sbps),np.std(sbps))
	print('BHS', BHS_standard(sbps))
	print('DBP stat : ',np.mean(dbps),np.std(dbps))
	print('BHS', BHS_standard(dbps))
	print('MAE ',np.mean(maes),np.std(maes))

	
	
	return 'SBP stat : {} +- {}'.format(np.mean(sbps),np.std(sbps)) + '\n' + 'BHS : {}'.format(BHS_standard(sbps)) + '\n' + 'DBP stat : {} +- {}'.format(np.mean(dbps),np.std(dbps)) + '\n' + 'BHS : {}'.format(BHS_standard(dbps)) + '\n' + 'MAE : {} +- {}'.format(np.mean(maes),np.std(maes))

