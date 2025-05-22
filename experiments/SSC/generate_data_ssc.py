
########################################################################################################################
# source code for downloading dataset found at https://compneuro.net/ (Cramer et al., 2020)
# Authors: Benjamin Cramer & Friedemann Zenke. Licensed under a Creative Commons Attribution 4.0 International License
# https://creativecommons.org/licenses/by/4.0/ slightly modified
########################################################################################################################

import os
import urllib.request
import gzip, shutil
from tensorflow.keras.utils import get_file
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

cache_dir=os.path.expanduser("./data")
cache_subdir="hdspikes"
print("Using cache dir: %s"%cache_dir)

# The remote directory with the data files
base_url = "https://compneuro.net/datasets"

# Retrieve MD5 hashes from remote
response = urllib.request.urlopen("%s/md5sums.txt"%base_url)
data = response.read()
lines = data.decode('utf-8').split("\n")
file_hashes = { line.split()[1]:line.split()[0] for line in lines if len(line.split())==2 }

def get_and_gunzip(origin, filename, md5hash=None):
    gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
    hdf5_file_path=gz_file_path[:-3]
    if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
        print("Decompressing %s"%gz_file_path)
        with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path

# Download the Spiking Speech Commands (SSC) dataset
files = ["ssc_test.h5.gz", "ssc_train.h5.gz"]

for fn in files:
    origin = "%s/%s"%(base_url,fn)
    hdf5_file_path = get_and_gunzip(origin, fn, md5hash=file_hashes[fn])
    print(hdf5_file_path)

# At this point we can visualize some of the data
# import tables
# import numpy as np
# fileh = tables.open_file(hdf5_file_path, mode='r')
# units = fileh.root.spikes.units
# times = fileh.root.spikes.times
# labels = fileh.root.labels
# This is how we access spikes and labels
# index = 0
# print("Times (ms):", times[index])
# print("Unit IDs:", units[index])
# print("Label:", labels[index])
# # A quick raster plot for one of the samples
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(16,4))
# idx = np.random.randint(len(times),size=3)
# for i,k in enumerate(idx):
#     ax = plt.subplot(1,3,i+1)
#     ax.scatter(times[k],700-units[k], color="k", alpha=0.33, s=2)
#     ax.set_title("Label %i"%labels[k])
#     ax.axis("off")
# plt.show()

########################################################################################################################
# source for preproccessing:
# https://github.com/byin-cwi/Efficient-spiking-networks/blob/main/SSC/ssc_generate_dataset.py (Yin et al., 2021)
#
# MIT License
#
# Copyright (c) 2021 byin-cwi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
########################################################################################################################
import os

"""
The dataset is 48kHZ with 24bits precision
* 700 channels
* longest 1.s
* shortest 0.21s
"""


import tables
import numpy as np

files = ['data/ssc_test.h5', 'data/ssc_train.h5']

fileh = tables.open_file(files[0], mode='r')
units = fileh.root.spikes.units
times = fileh.root.spikes.times
labels = fileh.root.labels

# This is how we access spikes and labels
index = 0
print("Times (ms):", times[index],max(times[index]))
print("Unit IDs:", units[index])
print("Label:", labels[index])


def binary_image_readout(times,units,dt = 1e-3):
    img = []
    N = int(1/dt)
    for i in range(N):
        
        idxs = np.argwhere(times<=i*dt).flatten()

        vals = units[idxs]
        vals = vals[vals > 0]

        vector = np.zeros(700)
        vector[700-vals] = 1

        times = np.delete(times,idxs)
        units = np.delete(units,idxs)
        img.append(vector)
        
    return np.array(img)


def generate_dataset(file_name,output_dir,dt=1e-3):
    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    # This is how we access spikes and labels
    index = 0
    print("Number of samples: ",len(times))
    for i in range(len(times)):
        x_tmp = binary_image_readout(times[i], units[i],dt=dt)
        y_tmp = labels[i]
        output_file_name = output_dir+'ID:'+str(i)+'_'+str(y_tmp)+'.npy'
        np.save(output_file_name, x_tmp)
    print('done..')
    return 0


generate_dataset(files[0],output_dir='data/test/',dt=4e-3)

# generate_dataset(files[1],output_dir='data/valid/',dt=4e-3)

generate_dataset(files[2],output_dir='data/train/',dt=4e-3)


