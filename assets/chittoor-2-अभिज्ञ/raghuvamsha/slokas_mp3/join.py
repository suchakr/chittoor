#%%
import os
import glob
memorize = [2, 5, 7, 8, 9, 13, 16, 17, 18, 19, 20, 29, 31, 32, 33, 34, 36, 44, 55, 58, 59, 60, 61, 63, 64, 65, 68, 69, 70, 75, 77, 79, 80, 84, 89, 90, 92, 104]
mp3s = sorted(glob.glob('rv*.mp3'))
silence_mp3 = sorted(glob.glob('1000-ms-pluck*.mp3'))[0]

mem_mp3 = "\n".join([f"file '{mp3}'\nfile '{silence_mp3}'\n"*4  for i, mp3 in enumerate(mp3s) if (i+1) in memorize])
all_mp3 = "\n".join([f"file '{mp3}'\nfile '{silence_mp3}'\n" for i, mp3 in enumerate(mp3s) if 'all' not in mp3])
with open('mem.txt', 'w') as f: f.write(mem_mp3)
with open('all.txt', 'w') as f: f.write(all_mp3)

# ffmpeg -f concat -safe 0 -i all.txt -c copy rv12_all.mp3
# ffmpeg -f concat -safe 0 -i mem.txt -c copy rv12_mem.mp3


# %%
