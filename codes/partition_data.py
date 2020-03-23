import pickle
import h5py
import os
import sys
from tqdm import tqdm
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_random_episode_list(minThresh=2500, ratio=0.25):
	files = next(os.walk('processed_data'))[2]

	sbps_dict = {}
	dbps_dict = {}

	sbps_cnt = {}
	dbps_cnt = {}

	dbps_taken = {}
	sbps_taken = {}

	sbps = []
	dbps = []

	candidates = []

	lut = {}

	for fl in tqdm(files, desc='Browsing Files'):

		lines = open(os.path.join('processed_data', fl),
					 'r').read().split('\n')[1:-1]

		for line in tqdm(lines, desc='Reading Lines'):

			values = line.split(',')

			file_no = int(fl.split('_')[1])
			record_no = int(fl.split('.')[0].split('_')[2])
			episode_st = int(values[0])
			sbp = int(float(values[1]))
			dbp = int(float(values[2]))

			if(sbp not in sbps_dict):

				sbps_dict[sbp] = []
				sbps_cnt[sbp] = 0

			sbps_dict[sbp].append((file_no, record_no, episode_st))
			sbps_cnt[sbp] += 1

			if(dbp not in dbps_dict):
				dbps_dict[dbp] = []
				dbps_cnt[dbp] = 0

			dbps_dict[dbp].append((file_no, record_no, episode_st, sbp))
			dbps_cnt[dbp] += 1

	sbp_keys = list(sbps_dict)
	dbp_keys = list(dbps_dict)

	sbp_keys.sort()
	dbp_keys.sort()

	for dbp in tqdm(dbp_keys, desc='DBP'):

		cnt = min(int(dbps_cnt[dbp]*ratio), minThresh)

		for i in tqdm(range(cnt)):

			indix = np.random.randint(len(dbps_dict[dbp]))

			candidates.append(
				[dbps_dict[dbp][indix][0], dbps_dict[dbp][indix][1], dbps_dict[dbp][indix][2]])

			if(dbp not in dbps_taken):
				dbps_taken[dbp] = 0

			dbps_taken[dbp] += 1

			if(dbps_dict[dbp][indix][3] not in sbps_taken):
				sbps_taken[dbps_dict[dbp][indix][3]] = 0

			sbps_taken[dbps_dict[dbp][indix][3]] += 1

			if(dbps_dict[dbp][indix][0] not in lut):

				lut[dbps_dict[dbp][indix][0]] = {}

			if(dbps_dict[dbp][indix][1] not in lut[dbps_dict[dbp][indix][0]]):

				lut[dbps_dict[dbp][indix][0]][dbps_dict[dbp][indix][1]] = {}

			if(dbps_dict[dbp][indix][2] not in lut[dbps_dict[dbp][indix][0]][dbps_dict[dbp][indix][1]]):

				lut[dbps_dict[dbp][indix][0]][dbps_dict[dbp]
											  [indix][1]][dbps_dict[dbp][indix][2]] = 1

			dbps_dict[dbp].pop(indix)

	for sbp in tqdm(sbp_keys, desc='SBP'):

		if sbp not in sbps_taken:
			sbps_taken[sbp] = 0
			tqdm.write('null')

		cnt = min(int(sbps_cnt[sbp]*ratio), minThresh) - sbps_taken[sbp]

		for i in tqdm(range(cnt)):

			while len(sbps_dict[sbp]) > 0:

				try:
					indix = np.random.randint(len(sbps_dict[sbp]))
				except:
					tqdm.write(str(len(sbps_dict[sbp])))

				try:
					dumi = lut[sbps_dict[sbp][indix][0]][sbps_dict[sbp]
														 [indix][1]][sbps_dict[sbp][indix][2]]

				except:
					sbps_dict[sbp].pop(indix)
					continue

				candidates.append(
					[sbps_dict[sbp][indix][0], sbps_dict[sbp][indix][1], sbps_dict[sbp][indix][2]])

				sbps_taken[sbp] += 1

				sbps_dict[sbp].pop(indix)

				break

	sbps_dict = {}
	dbps_dict = {}

	sbps_cnt = {}
	dbps_cnt = {}

	sbps = []
	dbps = []

	lut = {}

	print(len(candidates))

	pickle.dump(candidates, open('candidates.p', 'wb'))

	sbp_keys = list(sbps_taken)
	dbp_keys = list(dbps_taken)

	sbp_keys.sort()
	dbp_keys.sort()

	for sbp in sbp_keys:
		sbps.append(sbps_taken[sbp])

	for dbp in dbp_keys:
		dbps.append(dbps_taken[dbp])

	plt.figure()

	plt.subplot(2, 1, 1)
	plt.bar(sbp_keys, sbps)
	plt.title('SBP')

	plt.subplot(2, 1, 2)
	plt.bar(dbp_keys, dbps)
	plt.title('DBP')

	plt.show()

	print(np.sum(sbps), np.sum(dbps))


def partition_data(candidates, mode):

	odd = 1

	if(mode < 0):
		mode = -mode
		odd = 0

	f = h5py.File('./raw_data/Part_{}.mat'.format(mode), 'r')

	fs = 125
	t = 10
	samples_in_episode = round(fs * t)
	ky = 'Part_' + str(mode)

	for indix in tqdm(range(odd, len(candidates), 2)):

		if(candidates[indix][0] != mode):
			continue

		record_no = int(candidates[indix][1])
		episode_st = int(candidates[indix][2])

		ppg = []
		abp = []

		for j in tqdm(range(episode_st, episode_st+samples_in_episode)):

			ppg.append(f[f[ky][record_no][0]][j][0])
			abp.append(f[f[ky][record_no][0]][j][1])

		pickle.dump(np.array(ppg), open(
			os.path.join('ppgs', '{}.p'.format(indix)), 'wb'))
		pickle.dump(np.array(abp), open(
			os.path.join('abps', '{}.p'.format(indix)), 'wb'))

def random_data(mode):

	odd = 1

	if(mode < 0):
		mode = -mode
		odd = 0

	f = h5py.File('./raw_data/Part_{}.mat'.format(mode), 'r')

	fs = 125
	t = 10
	samples_in_episode = 1026

	ky = 'Part_' + str(mode)

	for record_no in tqdm(range(odd,3000,2)):

		up_lim = len(f[f[ky][record_no][0]]) - samples_in_episode - 1

		if(up_lim<=0):
			continue

		data = []

		for trial in tqdm(range(13)):        
		
			episode_st = np.random.randint(up_lim)

			ppg = []
			abp = []

			for j in tqdm(range(episode_st, episode_st+samples_in_episode)):

				ppg.append(f[f[ky][record_no][0]][j][0])
				abp.append(f[f[ky][record_no][0]][j][1])

			data.append({'ppg':np.array(ppg),'abp':np.array(abp)})
		
		
		pickle.dump(data, open(os.path.join('random_indpndt','{}_{}_{}.p'.format(mode,record_no,episode_st)), 'wb'))
			



def merge_results():

	files = next(os.walk('abps'))[2]

	print(files)

	data = []

	for fl in tqdm(files):

		abp = pickle.load(open(os.path.join('abps',fl),'rb'))
		ppg = pickle.load(open(os.path.join('ppgs',fl),'rb'))

		data.append([abp, ppg])

		#data.append({ 'abp':abp , 'ppg':ppg })

	
	f = h5py.File('data_big.hdf5', 'w')
	dset = f.create_dataset('data', data=data)

	#pickle.dump(data,open('data_big.p','wb'))

def view_files_stat():

	dt = pickle.load(open(os.path.join('data','meta0.p'),'rb'))
	max_ppg = dt['max_ppg']
	min_ppg = dt['min_ppg']
	max_abp = dt['max_abp']
	min_abp = dt['min_abp']

	sbps = []
	dbps = []
	maps = []

	files = next(os.walk('abps'))[2]

	print(files)

	

	for fl in tqdm(files):

		abp = pickle.load(open(os.path.join('abps',fl),'rb'))
		ppg = pickle.load(open(os.path.join('ppgs',fl),'rb'))
		
		maps.append(np.mean(abp))
		dbps.append(np.min(abp))
		sbps.append(np.max(abp))

	print(np.min(maps),np.max(maps),np.mean(maps),np.std(maps))
	print(np.min(dbps),np.max(dbps),np.mean(dbps),np.std(dbps))
	print(np.min(sbps),np.max(sbps),np.mean(sbps),np.std(sbps))


def getTotalLength():

	tot = 0

	for i in range(1,5):

		f = h5py.File('./raw_data/Part_{}.mat'.format(i), 'r')

		ky = 'Part_' + str(i)

		for record_no in tqdm(range(3000)):

			lenn = len(f[f[ky][record_no][0]])

			tot += lenn


	print(tot/125)



def main():

	getTotalLength()

	return

	view_files_stat()

	return
	
	#get_random_episode_list()

	#candidates = pickle.load(open('./candidates.p', 'rb'))
	
	mode = int(sys.argv[1])
	#mode = int(input('mode = '))

	#partition_data(candidates, mode)

	#merge_results()

	#mode = int(input('Mode = '))
	random_data(mode)


if __name__ == '__main__':
	main()
