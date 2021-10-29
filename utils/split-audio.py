#%%
import pandas as pd
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment
import IPython.display

import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
# #%%
# for f in glob.glob("../assets/*2*/rag*/*10.wav") :
# 	print(f)
# 	x, sr = librosa.load(f)
# 	# librosa.output.write_wav(f.replace(".mp3", ".wav"), y, sr)
# 	X = librosa.stft(x)
# 	Xdb = librosa.amplitude_to_db(abs(X))
# 	plt.figure(figsize=(14, 5))
# 	# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
# 	#If to pring log of frequencies  
# 	librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
# 	# plt.colorbar()
# 	plt.show()

# # %%

# import librosa.display
# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(x, sr=sr)

# %%
# suppress warnings
import warnings
import time
warnings.filterwarnings('ignore')
t0 = time.time()
fn = glob.glob("../assets/*2*/rag*/*10.wav")[0]
print(fn)
x, sr = librosa.load(fn)
print(f"Time to load {time.time() - t0}, {sr=}  {x.shape=}")

#%%
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
 
#%%
n=0
f = int(3*x.shape[0]//3)
a=int(n*f)
b=a+f
orig = x[a:b]
orig = pd.Series(orig)
abs_orig = pd.Series(np.abs(x[a:b]))
valleys = abs_orig.apply(lambda x: 0 if x < 0.08 else 1)

num_samples = 2*sr//3
silences=[0]
silences_modes=[.01]
t0 = time.time()
for i in range(0, len(valleys), num_samples):
	sound_clip = valleys[i:i+num_samples]
	binned_sound_clip = pd.Series([ x.mid for x in pd.cut(abs_orig[i:i+num_samples],10)])
	if sound_clip.mean() < 4e-10:	
		print(f"{int (time.time() - t0)} secs - {i:7d}  samples - {binned_sound_clip.sum():0.3f} sum - {binned_sound_clip.mean():0.3f} mean - {binned_sound_clip.mode()[0]:0.3f} mode")
		silences.append(i)
		silences_modes.append(binned_sound_clip.mode()[0])
#%%
t0 = time.time()
fig, axs = plt.subplots(2, 1, figsize=(24, 10))
axs = axs.flatten()
g = 0
valleys.plot(ax=axs[g], title='Valley'); g+= 1
orig.plot(ax=axs[g], title="Original") ; g += 1

# ax=axs[g-1].plot(silences, [0.5]*len(silences), 'ro')
for gg in range(1,3):
	ax=axs[g-gg].plot(
		silences, 
		[ x/len(silences) for x in range(len(silences))],
		'yo', markersize=10)
	ax=axs[g-gg].plot(
		silences, 
		100*np.array(silences_modes),
		'r*', markersize=10)
# silences
plt.show()
print(f"Time to load {time.time() - t0}, {sr=}  {orig.shape=}")

#%%
df = pd.DataFrame({ 
	'lo': silences[:-1], 
	'hi' : silences[1:],
	'scores':[ 100*x for x in silences_modes[1:] ]})
df = df[df.scores < 1.6]
# df['span'] = (df.hi - df.lo)/1e6
df['duration'] = (df.hi - df.lo)/sr
slokas = df[ [ 12 < x < 20 for x in df.duration ] ]
slokas
#%%
import soundfile as sf
iterrows = slokas.iterrows()
rags = list(orig)
for i, row in iterrows:
	wav = rags[int(row.lo):int(row.hi)]
	wav_fn = f"arag_10_{i:02d}.wav"
	print(wav_fn)
	sf.write(wav_fn, wav, sr)
# music.shape , sr2
#%%
hills = valleys.apply(lambda x: 1 if x == 0 else 0)
hills.plot(ax=axs[g], title='Hills'); g+= 1

#%%

# diff = orig.diff()
# diff.plot(ax=axs[g], title="Difference"); g+= 1 

# rolling_max = orig.rolling(window=1000).max()
# rolling_max.plot(ax=axs[g], title='Rolling max'); g+= 1


#%%
rolling_min = pd.Series(x[a:b]).rolling(window=1000).min()
pd.Series(rolling_min).plot(ax=axs[3], title='Rolling min')
pd.Series(rolling_min*rolling_max).plot(ax=axs[4], title='Rolling')
# rolling_mode = pd.Series(x[a:b]).rolling(window=10).apply(lambda x: x.mode()[0])
# pd.Series(rolling_mode).plot(ax=axs[2]) 

#%%
sig = x[a:b]
fft = np.fft.fft(sig)
fft_freq = np.fft.fftfreq(x.shape[0])
pd.Series(fft.real).plot(ax=axs[3]) 
pd.Series(fft.imag).plot(ax=axs[4]) 




# %%
