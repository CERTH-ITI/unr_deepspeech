#!/usr/bin/env python3
from __future__ import print_function

from sys import byteorder
from array import array
from struct import pack
import os
import pyaudio
import wave
import rospy
from unr_deepspeech.srv import *
from std_msgs.msg import String
from rospkg import RosPack
import audioop
import time
import noisereduce as nr
import librosa
import numba
import scipy
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

global rate, device

THRESHOLD = 3500
CHUNK_SIZE = 1000
FORMAT = pyaudio.paInt16
WAV_PATH = '/home/userofpc/ros_workspaces/hdetection_ws/src/unr_deepspeech/data/used_recording.wav'
SAVE_PATH = '/home/userofpc/ros_workspaces/hdetection_ws/src/unr_deepspeech/data/'
KEEP_WAV = True
CTRR = 0

# SUB FUNCTIONS
def noise_reduction():
    "Function for Noise reduction with FFT"
    # load data
    rate, data = wavfile.read(WAV_PATH)
    scipy.io.wavfile.write(SAVE_PATH+'_recording_noisy', rate, np.asarray(data, dtype=np.int16))

    data = np.ndarray.astype(data,float)

    # select section of data that is noise (ASSUMING that the whole recording contains surround noise)
    noisy_part = data[0:len(data)]

    # perform noise reduction
    reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=False, n_fft=2048, n_std_thresh=1 )
    reduced_noise = normalize(reduced_noise)
    scipy.io.wavfile.write(SAVE_PATH+'recording_noise_removed', rate, np.asarray(reduced_noise, dtype=np.int16))
    scipy.io.wavfile.write(WAV_PATH, rate, np.asarray(reduced_noise, dtype=np.int16))

def is_silent(sound_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(sound_data) < THRESHOLD

def normalize(sound_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in sound_data)

    r = array('h')
    for i in sound_data:
        r.append(int(i*times))

    return r

def trim(sound_data):
    "Trim the blank spots at the start and end"
    def _trim(sound_data):
        sound_started = False
        r = array('h')

        for i in sound_data:
            if not sound_started and abs(i)>THRESHOLD:
                sound_started = True
                r.append(i)

            elif sound_started:
                r.append(i)
        return r

    # Trim to the left
    sound_data = _trim(sound_data)

    # Trim to the right
    sound_data.reverse()
    sound_data = _trim(sound_data)
    sound_data.reverse()
    return sound_data

def add_silence(sound_data, seconds, rate):
    "Add silence to the start and end of 'sound_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*rate))])
    r.extend(sound_data)
    r.extend([0 for i in range(int(seconds*rate))])
    return r

# MAIN FUNCTIONS
def record(rate, device):
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """

    # This sets up a pyaudio port
    p = pyaudio.PyAudio() 

    # Following comes a setup of a Stream to "play" or "record" audio. 
    stream = p.open(format=FORMAT, channels=1, rate=rate, input=True, output=True, frames_per_buffer=CHUNK_SIZE, input_device_index=device)

    num_silent = 0
    sound_started = False

    r = array('h')

    while True:
        # little endian, signed short
        sound_data = array('h', stream.read(CHUNK_SIZE, exception_on_overflow=False))

        if byteorder == 'big':
            sound_data.byteswap()
            
        r.extend(sound_data)
        
        silent = is_silent(sound_data)
        
        if silent and sound_started:
            if num_silent==0:
                print("~~~Countdown started...~~~")
                num_silent += 1
            else:
                num_silent += 1

        elif not silent and not sound_started:
            print("Voice passed Threshold!!")
            sound_started = True

        if sound_started and num_silent > 50:
            break

    sample_width = p.get_sample_size(FORMAT)

    stream.stop_stream()
    stream.close()
    
    p.terminate()

    r = normalize(r)
    # r = trim(r)
    r = add_silence(r, 0.5, rate)

    return sample_width, r

def record_to_file(path, rate, device):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record(rate, device)

    if rate != 16000:
        data_16GHz = audioop.ratecv(data, sample_width, 1, rate, 16000, None)[0]
    else:
        data_16GHz = pack('<' + ('h'*len(data)), *data)
    
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(16000)
    wf.writeframes(data_16GHz)
    wf.close()


def call_deepspeech_transcription_service(filename):
    rospy.wait_for_service('listen')
    try:
        listener = rospy.ServiceProxy('listen', Listen)
        resp1 = listener(filename)
        return resp1.prediction
    except rospy.ServiceException as e:
        print ("Service call failed: %s"%e)

def record_callback(keyword):

    trigger = keyword.data
    print(trigger)
    record_audio_flag = rospy.get_param('/unr_deepspeech/record_flag')

    if trigger == "robot" and record_audio_flag==False :
        
        rospy.set_param('/unr_deepspeech/record_flag', param_value=True)
        record_audio_flag = rospy.get_param('/unr_deepspeech/record_flag')

        while record_audio_flag==True:

            # Wav file Recording        ~1~
            print("Ready to record")
            try:
                record_to_file(WAV_PATH, rate=rate, device=device)
            except:
                print("Error transcribing audio. Check your audio device index.")
                sys.exit(1)

            # Perform Noise Reduction using FFT     ~2~
            noise_reduction()

            # Use the new wav file to Transcribe speech     ~3~
            print("Transcribing speech...")
            deep_speech_output = call_deepspeech_transcription_service(WAV_PATH)
            print("\nText: " + deep_speech_output)
        
            # Clean the latest wav file     ~4~
            keep_wav = rospy.get_param("/unr_deepspeech/keep_wav", KEEP_WAV)
            if not keep_wav:
                os.remove(WAV_PATH)
            
            # calibrate_flag = rospy.get_param('/calibration_flag')


def main():
    p = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    numDevices = info.get('deviceCount')

    device = -1
    rate = 48000
      
    for i in range(0, numDevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                if p.get_device_info_by_host_api_device_index(0, i).get('name') == 'default':
                    device = i
                    print("Using device {}".format(i))
                    break
    if device == -1:
        print("Unable to find default device. Here are the available audio devices: ")
        for i in range(0, numDevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
        sys.exit(1)

    return rate,device

if __name__ == "__main__":

    rate, device = main()
    rospy.init_node("unr_deepspeech_client")
    rospy.set_param('/unr_deepspeech/record_flag', param_value=False)
    deepspeech_client_sub = rospy.Subscriber("kws_data", String, record_callback)
    rospy.spin()
