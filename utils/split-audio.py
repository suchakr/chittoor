#%%
import pandas as pd
import numpy as np
import librosa
import librosa.display
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
#%%
for f in glob.glob("../assets/*2*/rag*/*10.mp3") :
	print(f)
	x, sr = librosa.load(f)
	# librosa.output.write_wav(f.replace(".mp3", ".wav"), y, sr)
	X = librosa.stft(x)
	Xdb = librosa.amplitude_to_db(abs(X))
	plt.figure(figsize=(14, 5))
	librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
	#If to pring log of frequencies  
	# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
	plt.colorbar()
	plt.show()


	
# %%
%matplotlib inline

import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr=sr)

# %%
y.shape

# %%
librosa.display.waveplot(y[:10000*300], sr=sr)

# %%
X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
#If to pring log of frequencies  
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
# %%
