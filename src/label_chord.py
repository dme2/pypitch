from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import
import label_wav as lw

import pyaudio
import os
import wave

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#set parameters
RECORDING = True
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3

FLAGS = None

#initialize
p = pyaudio.PyAudio()

labels_list = "/tmp/chord_commands_train/conv_labels.txt"
wav_file_path = "/Users/dave/tensorflow/tensorflow/examples/speech_commands/temp.wav"

def listen(graph, labels,wav_file_path):
    frames = []
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    for i in range(0, int(RATE/CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    WAVE_OUTPUT_FILENAME = "temp.wav"
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    chord = lw.label_wav(wav_file_path, labels, graph,'wav_data:0', 'labels_softmax:0',1)
    return str(chord)
    


def listen_label(graph, labels,wav_file_path):
    #listen for 3s
    RECORDING = True
    while(RECORDING):
        inp = input("standby: ")
        if inp == 'r':
            print ("\n*listening")
            frames = []
            #open stream
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
            for i in range(0, int(RATE/CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("the chord is: ")    
            stream.stop_stream()
            stream.close()
            #write out file
            WAVE_OUTPUT_FILENAME= "temp.wav"
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            chord = lw.label_wav(wav_file_path,labels,
                                 graph, 'wav_data:0', 'labels_softmax:0',1)
            
            return chord

        elif inp == 'q':
            print("\n*exiting")
            RECORDING = False
            p.terminate()
        
#if __name__=='__main__':
 #   listen_label(graph='/tmp/my_graph.pb',labels=labels_list)
    
