"""
	Process the data downloaded from original source
"""

import h5py
import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def process_data():
	"""
		Extracts the SBP and DBP values of 10 seconds long episodes
		while taking new episodes 5 seconds apart
		and stores them as .csv files

		This function is likely to take 6-7 days to run on a Intel Core i7-7700 CPU
	"""

	fs = 125								# sampling frequency
	t = 10									# length of ppg episodes
	dt = 5									# step size of taking the next episode
	
	samples_in_episode = round(fs * t)		# number of samples in an episode
	d_samples = round(fs * dt)				# number of samples in a step

	try:									# create the processed_data directory
		os.makedirs('processed_data')
	except Exception as e:
		print(e)

	for k in range(1,5):					# process for the 4 different parts of the data

		print("Processing file part {} out of 4".format(k))

		f = h5py.File(os.path.join('raw_data','Part_{}.mat'.format(k)), 'r')		# loads the data

		ky = 'Part_' + str(k)														# key 

		for i in tqdm(range(len(f[ky])),desc='Reading Records'):					# reading the records

			signal = []												# ppg signal
			bp = []													# abp signal

			output_str = '10s,SBP,DBP\n'							# starting text for a new csv file

			for j in tqdm(range(len(f[f[ky][i][0]])),desc='Reading Samples from Record {}/3000'.format(i+1)):	# reading samples from records
				
				signal.append(f[f[ky][i][0]][j][0])					# ppg signal
				bp.append(f[f[ky][i][0]][j][1])						# abp signal

			for j in tqdm(range(0,len(f[f[ky][i][0]])-samples_in_episode, d_samples),desc='Processing Episodes from Record {}/3000'.format(i+1)):	# computing the sbp and dbp values
				
				sbp = max(bp[j:j+samples_in_episode])		# sbp value
				dbp = min(bp[j:j+samples_in_episode])    	# dbp value

				output_str += '{},{},{}\n'.format(j,sbp,dbp)	# append to the csv file


			fp = open(os.path.join('processed_data','Part_{}_{}.csv'.format(k,i)),'w')		# create the csv file
			fp.write(output_str)															# write the csv file
			fp.close()																		# close the csv file


def observe_processed_data():
	"""
		Observe the sbp and dbps of the 10s long episodes
	"""

	files = next(os.walk('processed_data'))[2]

	sbps = []
	dbps = []

	for fl in tqdm(files,desc='Browsing through Files'):

		lines = open(os.path.join('processed_data',fl),'r').read().split('\n')[1:-1]

		for line in tqdm(lines,desc='Browsing through Episodes from File'):

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

	plt.show()


def downsample_data(minThresh=2500, ratio=0.25):
	"""
	Downsamples the data based on the scheme proposed in the manuscript
	
	Keyword Arguments:
		minThresh {int} -- maximum number of episoeds (default: {2500})
		ratio {float} -- ratio of total signals of certain bin to take (default: {0.25})
	"""

	
	files = next(os.walk('processed_data'))[2]		# load all csv files

	sbps_dict = {}									# dictionary to store sbp and dbp values
	dbps_dict = {}

	sbps_cnt = {}									# dictionary containing count of specific sbp and dbp values
	dbps_cnt = {}

	dbps_taken = {}									# dictionary containing count of specific sbp and dbp taken
	sbps_taken = {}

	sbps = []										# list of sbps and dbps
	dbps = []

	candidates = []									# list of candidate episodes

	lut = {}										# look up table

	for fl in tqdm(files, desc='Browsing Files'):		# iterating over the csv files

		lines = open(os.path.join('processed_data', fl), 'r').read().split('\n')[1:-1]	# fetching the episodes

		for line in tqdm(lines, desc='Reading Episodes'):		# iterating over the episodes	

			values = line.split(',')

			file_no = int(fl.split('_')[1])							# id of the file
			record_no = int(fl.split('.')[0].split('_')[2])			# id of the record
			episode_st = int(values[0])								# start of the episode
			sbp = int(float(values[1]))								# sbp of that episode
			dbp = int(float(values[2]))								# dbp of that episode

			if(sbp not in sbps_dict):			# new sbp found

				sbps_dict[sbp] = []				# initialize
				sbps_cnt[sbp] = 0

			sbps_dict[sbp].append((file_no, record_no, episode_st))		# add the file, record and episode info
			sbps_cnt[sbp] += 1											# increment

			if(dbp not in dbps_dict):			# new dbp found

				dbps_dict[dbp] = []				# initialize
				dbps_cnt[dbp] = 0

			dbps_dict[dbp].append((file_no, record_no, episode_st, sbp))	# add the file, record and episode info
			dbps_cnt[dbp] += 1												# increment

	sbp_keys = list(sbps_dict)				# all the different sbp values
	dbp_keys = list(dbps_dict)				# all the different dbp values

	sbp_keys.sort()					# sorting the sbp values
	dbp_keys.sort()					# sorting the dbp values

	for dbp in tqdm(dbp_keys, desc='DBP Binning'):		# iterating through the dbp values

		cnt = min(int(dbps_cnt[dbp]*ratio), minThresh)		# how many episodes of this dbp to take

		for i in tqdm(range(cnt), desc='Picking Random Indices'):		

			indix = np.random.randint(len(dbps_dict[dbp]))		# picking a random index

			candidates.append([dbps_dict[dbp][indix][0], dbps_dict[dbp][indix][1], dbps_dict[dbp][indix][2]])	# add the file, record and episode info in the candidates list

			if(dbp not in dbps_taken):					# this dbp has not been taken
				dbps_taken[dbp] = 0						# initialize

			dbps_taken[dbp] += 1						# increment

			if(dbps_dict[dbp][indix][3] not in sbps_taken):		# checking if the sbp of that episode has been taken or not
				sbps_taken[dbps_dict[dbp][indix][3]] = 0		# initialize

			sbps_taken[dbps_dict[dbp][indix][3]] += 1			# increment

			if(dbps_dict[dbp][indix][0] not in lut):			# this file is not in look up table

				lut[dbps_dict[dbp][indix][0]] = {}				# add the file in look up table

			if(dbps_dict[dbp][indix][1] not in lut[dbps_dict[dbp][indix][0]]):	# this record is not in look up table

				lut[dbps_dict[dbp][indix][0]][dbps_dict[dbp][indix][1]] = {}	# add the record in look up table

			if(dbps_dict[dbp][indix][2] not in lut[dbps_dict[dbp][indix][0]][dbps_dict[dbp][indix][1]]):	# this episode is not in look up table

				lut[dbps_dict[dbp][indix][0]][dbps_dict[dbp][indix][1]][dbps_dict[dbp][indix][2]] = 1		# add this episode in look up table

			dbps_dict[dbp].pop(indix)		# remove this episode, so that this episode is not randomly selected again

	for sbp in tqdm(sbp_keys, desc='SBP Binning'):		# iterating on the sbps 

		if sbp not in sbps_taken:			# this sbp has not yet been taken
			sbps_taken[sbp] = 0				# initialize

		cnt = min(int(sbps_cnt[sbp]*ratio), minThresh) - sbps_taken[sbp]		# how many episodes of this sbp to take, removed the count already included during dbp based binning

		for i in tqdm(range(cnt), desc='Picking Random Indices'):		# iterate over how many episodes to take

			while len(sbps_dict[sbp]) > 0:					# while there are some episodes with that sbp left

				try:
					indix = np.random.randint(len(sbps_dict[sbp]))		# picking a random episode
				except:
					pass

				try:								# see if that episode is contained in the look up table
					dumi = lut[sbps_dict[sbp][indix][0]][sbps_dict[sbp][indix][1]][sbps_dict[sbp][indix][2]]	
				except:
					sbps_dict[sbp].pop(indix)	
					continue

				candidates.append([sbps_dict[sbp][indix][0], sbps_dict[sbp][indix][1], sbps_dict[sbp][indix][2]])	# add new candidate

				sbps_taken[sbp] += 1								# increment

				sbps_dict[sbp].pop(indix)							# remove that episode

				break												# repeat the process

	sbps_dict = {}			# garbage collection
	dbps_dict = {}

	sbps_cnt = {}			# garbage collection
	dbps_cnt = {}

	sbps = []				# garbage collection
	dbps = []

	lut = {}				# garbage collection

	print('Total {} episodes have been selected'.format(len(candidates)))	

	pickle.dump(candidates, open('candidates.p', 'wb'))		# save the candidates

	'''
		plotting the downsampled episodes
	'''

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

	

def extract_episodes(candidates):
	"""
		Extracts the episodes from the raw data


		This function is likely to take 3-4 days to run on a Intel Core i7-7700 CPU
	"""

	try:								# making the necessary directories
		os.makedirs('ppgs')
	except Exception as e:
		print(e)

	try:
		os.makedirs('abps')
	except Exception as e:
		print(e)

	for k in tqdm(range(1,5), desc='Reading from Files'):				# iterating throug the files

		f = h5py.File('./raw_data/Part_{}.mat'.format(k), 'r')

		fs = 125																# sampling frequency
		t = 10																	# length of ppg episodes			
		samples_in_episode = round(fs * t)										# number of samples in an episode
		ky = 'Part_' + str(k)													# key

		for indix in tqdm(range(len(candidates)), desc='Reading from File {}/4'.format(k)):		# iterating through the candidates

			if(candidates[indix][0] != k):					# this candidate is from a different file
				continue

			record_no = int(candidates[indix][1])			# record no of the episode
			episode_st = int(candidates[indix][2])			# start of that episode

			ppg = []										# ppg signal
			abp = []										# abp signal

			for j in tqdm(range(episode_st, episode_st+samples_in_episode), desc='Reading Episode Id {}'.format(indix)):	

				ppg.append(f[f[ky][record_no][0]][j][0])	# ppg signal
				abp.append(f[f[ky][record_no][0]][j][1])	# abp signal

			pickle.dump(np.array(ppg), open(os.path.join('ppgs', '{}.p'.format(indix)), 'wb'))		# saving the ppg signal
			pickle.dump(np.array(abp), open(os.path.join('abps', '{}.p'.format(indix)), 'wb'))		# saving the abp signal



def merge_episodes():
	"""
		Merges the extracted episodes
		and saves them as a hdf5 file
	"""

	try:									# creates the necessary directory
		os.makedirs('data')
	except Exception as e:
		print(e)

	files = next(os.walk('abps'))[2]				# all the extracted episodes

	np.random.shuffle(files)						# random shuffling, we perform the random shuffling now
													# so that we can split the data straightforwardly next step

	data = []										# initialize

	for fl in tqdm(files):

		abp = pickle.load(open(os.path.join('abps',fl),'rb'))			# abp signal
		ppg = pickle.load(open(os.path.join('ppgs',fl),'rb'))			# ppg signal

		data.append([abp, ppg])											# adding the signals


	
	f = h5py.File(os.path.join('data','data.hdf5'), 'w')				# saving the data as hdf5 file
	dset = f.create_dataset('data', data=data)


def main():
	process_data()
	observe_processed_data()
	downsample_data()
	candidates = pickle.load(open('./candidates.p', 'rb'))
	extract_episodes(candidates)
	merge_episodes()

if __name__ == '__main__':
	main()
