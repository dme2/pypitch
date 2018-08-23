import pyaudio
import os
import wave
import labelchord

#set parameters
RECORDING = True
CHUNK = 20148
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3

#initialize
p = pyaudio.PyAudio()

while(RECORDING):
    inp = input("standby")

    if inp == 'r':
        print ("\n*listening*")
        #record audio for 3s
        frames = []
        #open stream
        stream = p.open(format=FORMAT, channels=CHANNELS,rate=RATE,input=True,
                        frames_per_buffer=CHUNK)
        for i in range(0, int(RATE/CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        #write to file
        WAVE_OUTPUT_FILENAME = "temp.wav"
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        #send to cnn model
        label_wav("temp.wav",label_list, "path_to_graph")
        
        if inp == 'q':
            print ("\n*Exiting*")
            RECORDING = False
#clean up            
p.terminate()
