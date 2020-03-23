import pickle
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


'''def observe_data():

    files = next(os.walk('processed_data'))[2]

    sbps = []
    dbps = []

    for fl in tqdm(files):

        fp = pickle.load(open(os.path.join('processed_data',fl),'rb'))

        for dt in fp:

            sbps.append(dt['sbp'])
            dbps.append(dt['dbp'])

    plt.subplot(2,1,1)
    plt.hist(sbps,bins=60)
    plt.title('SBP')

    plt.subplot(2,1,2)
    plt.hist(dbps,bins=60)
    plt.title('DBP')

    plt.show()'''

def observe_processed_data():

    files = next(os.walk('processed_data'))[2]

    sbps = []
    dbps = []

    for fl in tqdm(files):

        lines = open(os.path.join('processed_data',fl),'r').read().split('\n')[1:-1]

        for line in tqdm(lines):

            values = line.split(',')
            
            sbp = int(float(values[1]))
            dbp = int(float(values[2]))

            sbps.append(sbp)
            dbps.append(dbp)


    plt.subplot(2,1,1)
    plt.hist(sbps,bins=180)
    plt.title('SBP')

    plt.subplot(2,1,2)
    plt.hist(dbps,bins=180)
    plt.title('DBP')

'''
def observe_long_data():

    files = next(os.walk('long_data'))[2]

    sbps = []
    dbps = []
    fs = 125
    t = 5
    dt = 2.5
    samples_in_episode = round(fs * t)
    d_samples = round(fs * dt)

    for fl in tqdm(files):

        fp = pickle.load(open(os.path.join('long_data',fl),'rb'))

        for i in tqdm(range(0,len(fp['abp'])-samples_in_episode, d_samples)):

            sbps.append(max(fp['abp'][i:i+samples_in_episode]))
            dbps.append(min(fp['abp'][i:i+samples_in_episode]))

    plt.subplot(2,1,1)
    plt.hist(sbps,bins=60)
    plt.title('SBP')

    plt.subplot(2,1,2)
    plt.hist(dbps,bins=60)
    plt.title('DBP')

    plt.show()

'''

def balance_bins(minThresh=1000, ratio=0.25):

    files = next(os.walk('processed_data'))[2]

    sbps_dict = {}
    dbps_dict = {}

    sbps = []
    dbps = []

    for fl in tqdm(files):

        lines = open(os.path.join('processed_data',fl),'r').read().split('\n')[1:-1]

        for line in tqdm(lines):

            values = line.split(',')
            
            sbp = int(float(values[1]))
            dbp = int(float(values[2]))

            if(sbp in sbps_dict):
                sbps_dict[sbp] += 1
            else:
                sbps_dict[sbp] = 1

            if(dbp in dbps_dict):
                dbps_dict[dbp] += 1
            else:
                dbps_dict[dbp] = 1

    sbp_keys = list(sbps_dict)
    dbp_keys = list(dbps_dict)

    sbp_keys.sort()
    dbp_keys.sort()
    
    for sbp in sbp_keys:
        sbps.append( min(int(sbps_dict[sbp]*ratio), minThresh ))
        

    for dbp in dbp_keys:
        dbps.append(min(int(dbps_dict[dbp]*ratio), minThresh ))

    plt.figure()

    plt.subplot(2,1,1)
    plt.bar(sbp_keys,sbps)
    plt.title('SBP')

    plt.subplot(2,1,2)
    plt.bar(dbp_keys,dbps)
    plt.title('DBP')

    print(np.sum(sbps),np.sum(dbps))



def main():
    observe_processed_data()

    balance_bins()

    plt.show()

if __name__ == '__main__':
    main()