from sklearn.ensemble import RandomForestClassifier as RFC
from play_audio import play_audio
import scipy.io.wavfile as wav
import numpy as np
import librosa
import glob
import pickle 

def result(clf,file):
	to_predict =[]
	sample_rate, X = wav.read(file)
	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=13)
	libceps = np.transpose(libceps)
	mfcc_delta = librosa.feature.delta(libceps)
	num_ceps = len(libceps)
        
	#mfcc mean
	mfcc_mean = np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0)
	mfcc_max = libceps.max(axis=0)
	mfcc_min = libceps.min(axis=0)
	mfcc_var = libceps.var(axis=0)
        
	#mfcc_delta
	mfcc_delta_mean = np.mean(mfcc_delta[int(num_ceps/10):int(num_ceps*9/10)], axis =0)
	mfcc_delta_max = mfcc_delta.max(axis=0)
	mfcc_delta_min = mfcc_delta.min(axis=0)
	mfcc_delta_var = mfcc_delta.var(axis=0)
        
	#mfcc_mean
	mfcc_mean_mean = np.mean(mfcc_mean)
	mfcc_mean_max = mfcc_mean.max()
	mfcc_mean_min = mfcc_mean.min()
	mfcc_mean_var = mfcc_mean.var()
        
	#mfcc_delta_mean
	mfcc_delta_mean_mean = np.mean(mfcc_delta_mean)
	mfcc_delta_mean_max = mfcc_delta_mean.max()
	mfcc_delta_mean_min = mfcc_delta_mean.min()
	mfcc_delta_mean_var = mfcc_delta_mean.var()
	feature = np.hstack((mfcc_mean,mfcc_max,mfcc_min,mfcc_var,mfcc_delta_mean,mfcc_delta_max,mfcc_delta_min,mfcc_delta_var,
                            mfcc_mean_mean,mfcc_mean_max,mfcc_mean_min,mfcc_mean_var,
                            mfcc_delta_mean_mean,mfcc_delta_mean_max,mfcc_delta_mean_min,mfcc_delta_mean_var))
	
	to_predict.append(feature)
	res = clf.predict(to_predict)
	print("Probability of the audio clip being happy: ",res)
	if res>=0.5:
		return "HAPPY !!!"
	else:
		return "SAD !!!"


saved = open("./trained.pickle","rb")
clf = pickle.load(saved)
saved.close()

emos = ['predict_happy.wav','predict_sad.wav']
for i in emos:
	play_audio(i)
	input()
	print(result(clf,i))
	input()
