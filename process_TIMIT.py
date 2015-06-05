"""
This is the 1st approach at trying to implement Speech recognition.
The first idea is to decouple Speech Recognition into several modules
    1.  Phoneme recognition
        extracting the phoneme out of the speech wave.
    2.  Word recogntion
        Linking the phonemes into words, probably a recurrent network
        that that decides when a word was prodcued.
    3.  Sentence recognition
        Linking words into sentences, probably antoher recurrent network

Here I'm only working on 1. Phoneme recognition

I'm transforming the problem of phoneme recognition into an image classification
one. I'm computing Mel Frequency Ceptral Coefficients (mfcc) during a time window
and the goal is to predict the phoneme from the image.

I will slice in time both the wav files and the annotated phoneme files. Each slice in
a wav file produces an mfcc that has to predict the phoneme.

Therefor the output of processing TIMIT corpus in getting it ready for a CNN will 
be an array of features (mfcc per slice of time) and a corresponding array of 
target phonemes

I am not worrying for the time being about the shapes of the arrays. Ultimately 
mfcc array will have to be 4D with (batches, channels, height, width) but I will 
do so in the timit_to_fuel.py converter
"""
import argparse
import numpy
import os
import os.path
import sys
import scipy.io.wavfile as wav
import re
import features as speech_features
import matplotlib.pyplot as plt
import pickle
import h5py
import time
from fuel.datasets.hdf5 import H5PYDataset
from fuel import config

import pdb

def wav_to_mfcc(wav_file, **kwarg):
    '''
    open the wav file and compute mfcc features
    for the time being I'm just using default parameters (25 ms fft window, 10 ms sliding window
    13 mfcc features. Later on I might customize this parameters)
    
    input:
    -----
        wav_file:   a full path to a wav file

    output:
    ------
        mfcc_feat:  2D ndarray with Mel Sepstrum coeficients in the y axis (13 by default)
                    and time in the x axis. 
                    By default time axis is such that each point corresponds to 25 ms of wav input
                    but bins are overlaping by 10 ms, so the following table is true:
                    bin#    start_time (ms) end_time (ms)
                    0       0               25
                    1       10              35
                    2       20              45
                    ...
                    n       n*10            n*10+25

        rate:       int, the sample rate

        count:      int, the number of samples

        kwarg:      dictionary to pass to features.mfcc

    Implementation note:
    -------------------
        I don't know why Scipy.io.wavfile can't open this wav files.
        I'm reading the files myself with two methods: read_header and read_data
        To verify that read_data is working ok, you can uncomment the print(line)
        statement and then check len(data), min(data), max(data) and compare those
        values from the ones in the header.

    '''
    def read_header(fid):
        fid.seek(0)
        for line in fid:
            #print(line)
            if line.startswith(b'sample_count'):
                sample_count = int(line.split()[-1])

            if line.startswith(b'sample_rate'):
                sample_rate = int(line.split()[-1])

            if line.startswith(b'end_head'):
                return sample_count, sample_rate

    def read_data(fid):
        fid.seek(1024)  # skip past the header, which is 1024 bits long.
        return numpy.fromstring(fid.read(), dtype=numpy.int16)

    with open(wav_file, 'rb') as f:
        (count, rate) = read_header(f)

        data = read_data(f)

    mfcc_feat = speech_features.mfcc(data, rate, **kwarg)
    return mfcc_feat, rate, count


def get_phoneme_target(phn_file, phoneme_dict, rate, count, winlen, winstep):
    '''
    open the phoneme file and convert it to a time series where the x axis is the same 
    as in the mfcc (25 ms bins sliding by 10 ms)
    instead of writing phonemes, write the int associated with it.
    Phonemes and integers as linked through a phoneme_dict that has phonemes as keys
    and their associated integer values as the dictionary values

    inputs:
    ------
        phn_file:   path to file with phoneme data.
                    each line in file is organized as: start_sample end_sample phoneme

        phoneme_dict:   dictionary that links each phoneme to a distinct integer
                        these integers are written into phoneme_target
                        the integer values start at 0, that's why I'm starting the
                        phoneme_dict with -1 (a value that never appears in the phoneme_dict)

        rate:       wav sample rate, needed to convert from wav's samples to mfcc's samples

        count:      number of samples in the wav file. Phoneme_target will have 
                    ceil((count - winlen)/winstep + 1) number of samples

        winlen:     length of window over which mfccs are computed

        winstep:    how far the time window slides from one chunck of wav file to the next
                    when computing mfccs

    outputs:
    -------
        phoneme_target: 1d numpy array with integers
                        each integer maps to a phoneme through phoneme_dict

    Implementation Note:
    -------------------
    TIMIT phoneme annotations are in samples.
    The wav's rate (along with some mfcc's parameters) defines the mfcc's time axis 
    In order to create the phoneme_target I need to convert from where the phoneme
    happens (in wav samples) to mfcc samples.

    with 'rate' number of wav samples in 1 sec then:
        winlen correspond to 'rate'*winlen wav samples, this is the width of a mfcc bin
        winstep correspond to 'rate'*winstep wav samples, distance between mfcc bins

    therefore the 'n' time bin in the mfcc, covering from [10ms*n, 10ms*n+25ms]
    translates into [rate*winstep*n, rate*winstep*n + rate*winlen] wav samples

    to convert from a wav sample to a mfcc sample I just do:
        round(wav_sample/(rate*winstep))

    '''
    # In the following line, -1 is a symbol that does not happen in my phoneme_dict
    # I'm not s:ure why I need to sum 1 to the number of items i'm creating but without it
    # phonee_target is always one short when compared to mfcc.shape[0]
    phoneme_target = numpy.ones(   
            numpy.ceil((count - rate*winlen)/(rate*winstep) + 1), dtype=numpy.int16) * -1

    #pdb.set_trace()
    with open(phn_file) as f:
        try:
            for line in f:
                tokens = line.split()
                start = numpy.round(int(tokens[0])/(rate*winstep))
                end = numpy.round(int(tokens[1])/(rate*winstep))
                # I'm constructing phoneme_dict (and word_dict) such that the values are
                # tuples. The 1st value in the tuple is always the int associated with
                # either the phoneme or the word. Even though the phoneme_dict value, is a
                # tuple with just one value (and I could get away with just storing the int)
                # I'm constructing them in this way because the word_dict does have other
                # properties (like the phoneme corresponding to the word) and I want this
                # function to be generic for both. 
                # Either I have to test here whether phoneme_dict[tokens[2]] is iterable
                # or make it always an iterable
                phoneme_target[start:end] = phoneme_dict[tokens[2]][0]
        except:
            raise ValueError('get_phoneme_target failed while processing {file}'.format(
                file=phn_file))
                
            #pdb.set_trace()
    return phoneme_target

def get_word_target(word_file, word_dict, rate, count):
    '''
    open the word file and convert it to a time series where the x axis is the same
    as in the mfcc (25 ms bins sliding by 10 ms)
    instead of writing phonemes, write the int associated with it.
    Phonemes and integers as linked through a phoneme_dict that has phonemes as keys
    and their associated integer values as the dictionary values
    
    This function is identical to phoneme_target
    '''
    return get_phoneme_target(word_file, word_dict, rate, count)

def get_phoneme_dict():
    '''
    open the file phonemes.txt and produce a dictionary with phonemes as keys and
    non-repeating integers as values

    the filephonemes was created by me from .../timit/doc/phonecod.doc
    '''

    symbol_re = re.compile('\s*(?P<symbol>[\w\-#]*).*')
    
    next_symbol_index = 0
    phoneme_dict = {}

    with open('phonemes.txt') as f:
        for line in f:
            if line[0]=='#':
                continue

            symbol = symbol_re.split(line)[1]

            # I'm making the value associated with symbol a tuple (instead of
            # just storing the next_symbol_indes) such that 
            # get_phoneme_target can be reused without modification as
            # get_word_target
            phoneme_dict[symbol] = (next_symbol_index,)
            next_symbol_index += 1

    return phoneme_dict

def get_word_dict():
    '''
    open the file doc/timitdic.txt and produce a dictionary with words as keys and tuples as 
    values. Each tuple is composed of two items, a unique integer (starting from 0 and 
    incrementing by 1) identifying the word and a list with the phonemes that compose the
    word
    '''

    symbol_re = re.compile(r'(?P<word>[\w\-\']*)\s*\/(?P<phonemes>.*)\/')
    
    next_word_index = 0
    word_dict = {}

    with open('timitdic.txt') as f:
        for line in f:
            if line[0]==';':
                continue

            word_phonemes = symbol_re.split(line)

            word = word_phonemes[1]
            phonemes = word_phonemes[2].split()
            
            word_dict[word] = (next_word_index, phonemes)
            next_word_index += 1

    return word_dict

def process_all_files(path, phoneme_dict, word_dict, results, **kwargs):
    '''
    recursively get into every folder under path until you find files with .phn, .wav, .wrd
    extensions.

    once those files are found, compute mfcc, phoneme_target and word_target

    append those to corresponding lists in results dic (results['mfcc'], 
    results['phoneme_target'], results['word_target'], results['file'])
    where results['file'] is just the full path to files common name (removing extension and 
    initial path .../TIMIT/timit)

    inputs:
    ------
        path:   when you call this function is the starting to descend the tree.
                it should be .../TIMIT/timit/ where the test and train folders exist
        
        phoneme_dict:   dictionary linking phonemes to unique integers (target for phoneme)
                        Integers are stored in a tuple
                        For example: phoneme_dict['b'] = (0,)

        word_dict:      dictionary linking words to tuples,
                        each tuple as a unite integer (target for the word) and
                        a list representing the word as a sequence of phonemes 
    '''
    #pdb.set_trace()
    if len(results['file'])>samples:
        return


    # the the list of folders below path
    folder_list = [os.path.join(path,folder) for folder in os.listdir(path)]
    folder_list = [folder for folder in folder_list if os.path.isdir(folder)]

    # .phn, .wav, .wrd files are at the end of the folder tree, when there are no more
    # folders to get into
    if folder_list:
        for folder in folder_list:
            process_all_files(folder, phoneme_dict, word_dict, results, **kwargs)
    else:
        phn_list = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('phn')]

        # process each file in phn_list but first make sure that .wav and .wrd files
        # also exist
        #pdb.set_trace()
        for phn_f in phn_list:
            wav_f = phn_f.rstrip('phn') + 'wav'
            wrd_f = phn_f.rstrip('phn') + 'wrd'

            if not os.path.isfile(phn_f) or not os.path.isfile(wav_f) or not os.path.isfile(wrd_f):
                raise ValueError('process_all_files found a potential phn file without either phn, wav or wrd file')
            #print('processing file ' + phn_f.rstrip('.phn'))

            # Get all the mfccs and phonemes in a wav/phn pair of files
            #pdb.set_trace()
            mfcc, rate, count = wav_to_mfcc(wav_f, **kwargs)
            phoneme_target = get_phoneme_target(phn_f, phoneme_dict, rate, count,
                    kwargs['winlen'], kwargs['winstep'])

            if mfcc.shape[0] != len(phoneme_target):
                pdb.set_trace()
                raise ValueError("mfcc and phoneme_target don't have the same number of time slices")

            #pdb.set_trace()
            # add each mfcc/phoneme independently (not as a sentence) to results
            results['mfcc'] += [mfcc[i,:] for i in range(mfcc.shape[0])]
            results['phoneme_target'] += phoneme_target.tolist()

            if len(results['mfcc']) != len(results['phoneme_target']):
                raise ValueError("""results lists associated with 'mfcc' and
                    'phoneme_target' are of different lengths""")

            """
            This might be useful if droping the idea of predicting phonemes from
            individual mfccs

            results['mfcc'].append(mfcc)

            results['phoneme_target'].append(
                    get_phoneme_target(phn_f, phoneme_dict, rate, count))
            results['word_target'].append(
                    get_word_target(wrd_f, word_dict, rate, count))

            """
            # remove everything up to (and including) 'timit' from path
            results['file'].append(
                    phn_f.rstrip('.phn').split('timit')[1])

def Fuel(path_out, results, phoneme_dict, word_dict):
    '''
    Save data in Fuel format
    inputs:
    ------
        path_out:   path to store the data

        results:    dictionary with 'mfcc', 'phoneme_target', 'word_target' keys
                    each associated value is a list

        phoneme_dict:   has phoneme as 'keys' and a tuple with unique integer values as 'values'

        word_dict:      has words as 'keys' and a two value tuple as 'values'
                        Each tuple is compossed of a unique word integer and how the
                        word is decompossed in phonemes
    '''
    f = h5py.File(os.path.join(path_out,'timit.hdf5'), mode='w')

    # mfccs, word_target, phoneme_target don't have constant size.
    # I'll have to import them as ragged objects.
    # phoneme/word_target are 1d, but mfccs are 2d.
    N = len(results['mfcc'])
    testN = len([s for s in results['file'] if s.startswith('/test')])
    
    #mfcc_dtype = h5py.special_dtype(vlen=results['mfcc'][0].dtype)     # resutls['mfcc'] is a list, grab the 1st element and poke its dtype
    #target_dtype = h5py.special_dtype(vlen=numpy.dtype(numpy.int16))
    
    mfcc = f.create_dataset('mfcc', (N,13), dtype=numpy.float)
    phoneme_target = f.create_dataset('phoneme_target', (N,1), dtype=numpy.int16)

    mfcc[...] = numpy.array(results['mfcc'])
    phoneme_target[...] = numpy.array(results['phoneme_target']).reshape(-1,1)

    mfcc.dims[0].label = 'batch'
    mfcc.dims[1].label = 'mfcc'
    phoneme_target.dims[0].label = 'batch'
    phoneme_target.dims[1].label = 'phoneme'

    #mfcc_shapes_labels = f.create_dataset(
    #        'mfcc_shapes_labels', (2,), dtype='S4')
    #target_shapes_labels = f.create_dataset(
    #        'target_shapes_labels', (1,), dtype='S4')

    #mfcc_shapes_labels[...] = [
    #        'mfcc'.encode('utf8'), 'time'.encode('utf8')]
    #target_shapes_labels[...] = [
    #        'time'.encode('utf8')]

    #mfcc.dims.create_scale(
    #        mfcc_shapes_labels, 'shape_labels')
    #word_target.dims.create_scale(
    #        target_shapes_labels, 'shape_labels')
    #phoneme_target.dims.create_scale(
    #        target_shapes_labels, 'shape_labels')

    #mfcc.dims[0].attach_scale(mfcc_shapes_labels)
    #word_target.dims[0].attach_scale(target_shapes_labels)
    #phoneme_target.dims[0].attach_scale(target_shapes_labels)
    
    split_dict = {
            'test': {
                #'word_target': (0, testN),
                'phoneme_target': (0, testN),
                'mfcc': (0, testN)},
            'train': {
                #'word_target': (testN,N),
                'phoneme_target': (testN,N),
                'mfcc': (testN,N)}
            }
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()

def phoneme_length(path, length_list):
    '''
    recursively get into every folder under path until you find files with .phn, .wav, .wrd
    extensions.

    once those files are found, compute the length of each phoneme in wav samples.
    
    Append result to length_list

    inputs:
    ------
        path:   when you call this function is the starting to descend the tree.
                it should be .../TIMIT/timit/ where the test and train folders exist
        
        phoneme_list:   list of times each phoneme is pronounced (in wav samples)
    '''

    if len(length_list)>=10000:
        return

    folder_list = [os.path.join(path,folder) for folder in os.listdir(path)]
    folder_list = [folder for folder in folder_list if os.path.isdir(folder)]

    # .phn, .wav, .wrd files are at the end of the folder tree, when there are no more
    # folders to get into
    if folder_list:
        for folder in folder_list:
            phoneme_length(folder, length_list)
    else:
        phn_list = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('phn')]

        for phn_f in phn_list:
            with open(phn_f) as f:
                for line in f:
                    tokens = line.split()
                    length_list.append(int(tokens[1])-int(tokens[0]))

if __name__ == '__main__':
    # Implement input parser
    path_out = os.path.join(config.data_path, 'timit')
    parser = argparse.ArgumentParser(description='get data from TIMIT into Fuel')
    parser.add_argument('path_in', action="store", type=str, help='path to timit/ folder')
    parser.add_argument('--path_out', action="store",
            default=path_out, type=str, help='name of output file')
    parser.add_argument('--hist', action="store_true",
            default=False, help='compute histogram of phoneme durations in wav units')
    parser.add_argument('--winlen', action="store",
            default=.015, type=float, help='defines the time window for FFTs (in seconds)')
    parser.add_argument('--winstep', action="store",
            default=0.1, type=float, help='defines how much the FFT window slides (in seconds)')
    parser.add_argument('--numcep', action="store",
            default=13, type=int, help='Number of ceptrum coeficients')
    parser.add_argument('--samples', action="store",
            default=numpy.inf, type=int, help="if given, will quit once 'samples' have been acquired")

    args = parser.parse_args()
    path_in = args.path_in
    path_out = args.path_out
    hist = args.hist
    global samples
    samples= args.samples

    kwargs = {'winlen':args.winlen, 'winstep':args.winstep, 'numcep':args.numcep}

    if not os.path.isdir(path_in):
        raise ValueError('{path} is not a correct path'.format(path=path_in))
    if not os.path.isdir(path_out):
        raise ValueError('{path} is not a correct path'.format(path=path_out))

    # input parser ends here

    # start processing data

    t0 = time.time()

    #pdb.set_trace()
    if hist:
        length_list = []
        phoneme_length(path_in, length_list)
        length_list = numpy.array(length_list)
        print('some stats about phoneme duration:')
        print('\tmin:\t', length_list.min())
        print('\tmean:\t', length_list.mean())
        print('\tmax:\t', length_list.max())
        print('\tstd:\t', length_list.std())
        y, x = numpy.histogram(length_list, bins=list(range(0,1500,10)))
        plt.plot(x[:-1],y)
        plt.show()

    phoneme_dict = get_phoneme_dict()
    word_dict = get_word_dict()
    with open(os.path.join(path_out, 'phoneme_dict.pkl'), 'wb') as f:
        pickle.dump(phoneme_dict, f)
    with open(os.path.join(path_out, 'word_dict.pkl'), 'wb') as f:
        pickle.dump(word_dict, f)

    for folder in ('test', 'train'):
        results = {
                'mfcc':[], 
                'phoneme_target':[],
                'word_target':[],
                'file':[]
                }

        process_all_files(
                os.path.join(path_in, folder),
                phoneme_dict,
                word_dict,
                results,
                **kwargs)

        
        # timit_to_fuel needs numpy arrays, not list
        results['mfcc'] = numpy.array(results['mfcc'])
        results['phoneme_target'] = numpy.array(results['phoneme_target'])

        with open(os.path.join(path_out, folder + '_mfcc.pkl'), 'wb') as f:
            pickle.dump(results['mfcc'], f)
        with open(os.path.join(path_out, folder + '_phonemes.pkl'), 'wb') as f:
            pickle.dump(results['phoneme_target'], f)

    t0 = time.time()-t0
    print("processing {samples} took {time}seconds".format(samples=samples, time=t0))

