from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_CHORD_LABEL = '_unknown_'
UNKNOWN_CHORD_INDEX = 1
RANDOM_SEED = 59185

def prepare_chords_list(wanted_chords):
    """
    input: 
    wanted_chords - list containing chord names
     
    return:
    list with silence and unknown labels added
    """
    lis = list(wanted_chords)
    return [SILENCE_LABEL, UNKNOWN_CHORD_LABEL] + lis

def which_set(filename, validation_percentage, testing_percentage):
    """
    keep the files in training, validation and testing consistent
    
    input:
    filename - file path
    validation_percentage - percent of data to be used for validation
    testing_percentage - percent of data to be used for testing
    """

    base_name = os.path.basename(filename)
    #ignore anything with no_hash in the filename 
    hash_name = re.sub(r'_nohash_.*$', '' , base_name)

    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    
    percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
    
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

def load_wav_file(filename):
   """
   load the audio file and return array of samples

   input: 
   filename - file path
   
   return:
   numpy array of normalized sample data
   """

   with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()

def save_wav_file(filename, wav_data, sample_rate):
      """
      save audio sample data to a .wav file

      input:
      filename - file path
      wav_data - 2d array of audio data
      sample_rate - samples per second
      """
      with tf.Session(graph=tf.Graph()) as sess:
          wav_filename_placeholder = tf.placeholder(tf.string, [])
          sample_rate_placeholder = tf.placeholder(tf.int32, [])
          wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
          wav_encoder = contrib_audio.encode_wav(wav_data_placeholder,
                                           sample_rate_placeholder)
          wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
          sess.run(
              wav_saver,
              feed_dict={
                  wav_filename_placeholder: filename,
                  sample_rate_placeholder: sample_rate,
                  wav_data_placeholder: np.reshape(wav_data, (-1, 1))
              })


class AudioProcessor(object):

    def __init__(self, data_dir, silence_percentage, unknown_percentage, wanted_chords,
                 validation_percentage, testing_percentage, model_settings):
    
        self.data_dir = data_dir
        self.prepare_data_index(silence_percentage, unknown_percentage, wanted_chords,
                                validation_percentage, testing_percentage)
        self.prepare_processing_graph(model_settings)

    def prepare_data_index(self, silence_percentage, unknown_percentage, wanted_chords,
                           validation_percentage, testing_percentage):
        """
        Prepare a list of smaples organized by set and label
        """
        
       # print (wanted_chords)
        random.seed(RANDOM_SEED)
        wanted_chords_index = {}
        for index, wanted_chords in enumerate(wanted_chords):
            wanted_chords_index[wanted_chords] = index + 2
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_chords = {}
        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in gfile.Glob(search_path):
            _, chord = os.path.split(os.path.dirname(wav_path))
            chord = chord.lower()
            all_chords[chord] = True
            set_index = which_set(wav_path, validation_percentage, testing_percentage)
            if chord in wanted_chords_index:
                self.data_index[set_index].append({'label': chord, 'file': wav_path})
            else:
                unknown_index[set_index].append({'label': chord, 'file': wav_path})
        #raise exception if no chords found
        if not all_chords:
            raise Exception('No .wavs found at ' + search_path)
        

      #  for index, wanted_chord in enumerate(wanted_chords):
       #     print (wanted_chord)
        #    if wanted_chord not in all_chords and wanted_chord != 'g':
                  #raise Exception('Expected to find ' + wanted_chord +
                   #             ' in labels but only found ' + ', '.join(all_chords.keys()))
            
        
        #load an arbitray file as the input for silence
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path
                    })
            #pick some uknonwns to add to each partition
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.chords_list = prepare_chords_list(wanted_chords)
        self.chord_to_index = {}
        for chord in all_chords:
            if chord in wanted_chords_index:
                self.chord_to_index[chord] = wanted_chords_index[chord]
            else:
                self.chord_to_index[chord] = UNKNOWN_CHORD_INDEX
        self.chord_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def prepare_processing_graph(self, model_settings):
        """
        Build a tensorflow graph
        
        creates a graph that loads a wave file, decodes it, scales the volume, shifts it in time
        calculates a spectrogram and builds MFCC fingerprint from that

        input:
        model_settings: info about model being trained
        """
        desired_samples = model_settings['desired_samples']
        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
        wav_decoder = contrib_audio.decode_wav(
             wav_loader, desired_channels=1, desired_samples=desired_samples)
         # Allow the audio sample's volume to be adjusted.
        self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
        scaled_foreground = tf.multiply(wav_decoder.audio,
                                   self.foreground_volume_placeholder_)
         # Shift the sample's start position, and pad any gaps with zeros.
        self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
        self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
        padded_foreground = tf.pad(
            scaled_foreground,
            self.time_shift_padding_placeholder_,
            mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground,
                                 self.time_shift_offset_placeholder_,
                                 [desired_samples, -1])

        self.background_data_placeholder_ = tf.placeholder(np.float32,                                             [desired_samples, 1])
        self.background_volume_placeholder_ = tf.placeholder(np.float32, [])
        background_mul = tf.multiply(self.background_data_placeholder_,
                                 self.background_volume_placeholder_)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
        spectrogram = contrib_audio.audio_spectrogram(
            background_clamp,
            window_size=model_settings['window_size_samples'],
            stride=model_settings['window_stride_samples'],
            magnitude_squared=True)
        self.mfcc_ = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=model_settings['dct_coefficient_count'])

    def get_size(self, model):
        return len(self.data_index[mode])

    def get_data(self, how_many, offset, model_settings, background_frequency,
                 background_volume_range, time_shift, mode, sess):
        
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
            # Data and labels will be populated and returned.
            data = np.zeros((sample_count, model_settings['fingerprint_size']))
            labels = np.zeros(sample_count)
            desired_samples = model_settings['desired_samples']
            
            pick_deterministically = (mode != 'training')
            # Use the processing graph we created earlier to repeatedly to generate the
            # final output sample data we'll use in training.
            for i in xrange(offset, offset + sample_count):
                # Pick which audio sample to use.
                if how_many == -1 or pick_deterministically:
                    sample_index = i
                else:
                    sample_index = np.random.randint(len(candidates))
                    sample = candidates[sample_index]
                # If we're time shifting, set up the offset for this sample.
                if time_shift > 0:
                    time_shift_amount = np.random.randint(-time_shift, time_shift)
                else:
                    time_shift_amount = 0
                if time_shift_amount > 0:
                    time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                    time_shift_offset = [0, 0]
                else:
                    time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                    time_shift_offset = [-time_shift_amount, 0]
                    input_dict = {
                        self.wav_filename_placeholder_: sample['file'],
                        self.time_shift_padding_placeholder_: time_shift_padding,
                        self.time_shift_offset_placeholder_: time_shift_offset,
                }
        # If we want silence, mute out the main sample but leave the background.
        if sample['label'] == SILENCE_LABEL:
            input_dict[self.foreground_volume_placeholder_] = 0
        else:
            input_dict[self.foreground_volume_placeholder_] = 1
        # Run the graph to produce the output audio.
        data[i - offset, :] = sess.run(self.mfcc_, feed_dict=input_dict).flatten()
        label_index = self.word_to_index[sample['label']]
        labels[i - offset] = label_index
        return data, labels

    def get_unprocessed_data(self, how_many, model_settings, mode):
       """Retrieve sample data for the given partition, with no transformations.

      Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      model_settings: Information about the current model being trained.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.

    Returns:
      List of sample data for the samples, and list of labels in one-hot form.
      """
       candidates = self.data_index[mode]
       if how_many == -1:
          sample_count = len(candidates)
       else:
          sample_count = how_many
       desired_samples = model_settings['desired_samples']
       words_list = self.words_list
       data = np.zeros((sample_count, desired_samples))
       labels = []
       with tf.Session(graph=tf.Graph()) as sess:
          wav_filename_placeholder = tf.placeholder(tf.string, [])
          wav_loader = io_ops.read_file(wav_filename_placeholder)
          wav_decoder = contrib_audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
          foreground_volume_placeholder = tf.placeholder(tf.float32, [])
          scaled_foreground = tf.multiply(wav_decoder.audio,
                                      foreground_volume_placeholder)
          for i in range(sample_count):
              if how_many == -1:
                  sample_index = i
              else:
                  sample_index = np.random.randint(len(candidates))
              sample = candidates[sample_index]
              input_dict = {wav_filename_placeholder: sample['file']}
              if sample['label'] == SILENCE_LABEL:
                  input_dict[foreground_volume_placeholder] = 0
              else:
                  input_dict[foreground_volume_placeholder] = 1
              data[i, :] = sess.run(scaled_foreground, feed_dict=input_dict).flatten()
              label_index = self.word_to_index[sample['label']]
              labels.append(words_list[label_index])
       return data, labels
