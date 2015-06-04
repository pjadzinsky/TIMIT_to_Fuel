"""
I'm modifying ~/miniconda/lib/pyton3.4/site-packages/fuel/converters/mnist.py

Original file is for the time being in
  ~/Documents/Development/python/projects/TIMIT_to_Fuel/TIMIT_to_Fuel/timit.py

I have made a link to this file from
  ~/miniconda/lib/python3.4/site-packages/fuel/converters/timit.py


The way I'm writing timit processing I'm outputing several pickle files

phoneme_dict.pkl    translates from phonemes to labels
word_dict.pkl       translates from words to labels
test_mfcc.pkl       testing features, list of ndarrays (each with shape (1, 1, numcep) )
test_phonemes.pkl   testing ground truth, list of integers
train_mfcc.pkl      training features, list of ndarrays (each with shape (1, 1, numcep) )
train_phonemes.pkl  training ground truth, list of integers
"""

import os
import struct

import h5py
import numpy
import pickle
import argparse
from fuel import config
import pdb

from fuel.converters.base import fill_hdf5_file, check_exists

# this will probably change
TRAIN_IMAGES = 'train_mfcc.pkl'
TRAIN_LABELS = 'train_phonemes.pkl'
TEST_IMAGES = 'test_mfcc.pkl'
TEST_LABELS = 'test_phonemes.pkl'

ALL_FILES = [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS]


@check_exists(required_files=ALL_FILES)
def convert_timit(directory, output_file):
    """Converts the TIMIT dataset to HDF5.

    Converts the TIMIT dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.TIMIT`. The converted dataset is
    saved as 'timit.hdf5'.

    This method assumes the existence of the following files:
    TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS defined above

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_file : str
        Where to save the converted dataset.

    """
    #pdb.set_trace()
    h5file = h5py.File(output_file, mode='w')

    with open(os.path.join(directory, TRAIN_IMAGES), 'rb') as f:
        train_features = pickle.load(f)
    with open(os.path.join(directory, TRAIN_LABELS), 'rb') as f:
        train_labels = pickle.load(f)
    with open(os.path.join(directory, TEST_IMAGES), 'rb') as f:
        test_features = pickle.load(f)
    with open(os.path.join(directory, TEST_LABELS), 'rb') as f:
        test_labels = pickle.load(f)

    # Force {train,test}_features to be 4D (batch, channel, height, width)
    if train_features.ndim==2:
        train_features = numpy.expand_dims(train_features, axis=1)
    if train_features.ndim==3:
        train_features = numpy.expand_dims(train_features, axis=1)
    if test_features.ndim==2:
        test_features = numpy.expand_dims(test_features, axis=1)
    if test_features.ndim==3:
        test_features = numpy.expand_dims(test_features, axis=1)
    # Force {train,test}_labels to be 2D (batch, index)
    if train_labels.ndim==1:
        train_labels = numpy.expand_dims(train_labels, axis=1)
    if test_labels.ndim==1:
        test_labels = numpy.expand_dims(test_labels, axis=1)

    data = (('train', 'features', train_features),
            ('train', 'targets', train_labels),
            ('test', 'features', test_features),
            ('test', 'targets', test_labels))

    fill_hdf5_file(h5file, data)
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'channel'
    h5file['features'].dims[2].label = 'height'
    h5file['features'].dims[3].label = 'width'
    h5file['targets'].dims[0].label = 'batch'
    h5file['targets'].dims[1].label = 'index'

    h5file.flush()
    h5file.close()

if __name__=='__main__':
    #pdb.set_trace()
    default_path = os.path.join(config.data_path, 'timit')

    parser = argparse.ArgumentParser(description="Convert pkl files into timit.hdf5 for fuel to process")
    parser.add_argument('--input_path', action="store", default=default_path, type=str, help="Pathway to pkl files")
    parser.add_argument('--output_file', action="store", default=os.path.join(default_path, 'timit.hdf5'),
            type=str, help="Where do you want the hdf5 file?")
    parser.add_argument('--clean', action="store_true", help="remove pkl files after generating hdf5")

    # if either input_path or output_path are default, I'll use the path in ~/.fuelrc

    args = parser.parse_args()
    
    convert_timit(args.input_path, args.output_file)

    if args.clean:
        pkl_files = [f for f in os.listdir(args.input_path) if f.endswith('pkl')]
        for f in pkl_files:
            os.remove(os.path.join(args.input_path, f))
