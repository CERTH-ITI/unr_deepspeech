#!/usr/bin/env python3
import speech_recognition as sr
from std_msgs.msg import String
import numpy as np
import pyaudio
from array import array
import time 
import rospy
from sys import byteorder
from statistics import mean

from unr_deepspeech.srv import *

CHUNK_SIZE = 1000
FORMAT = pyaudio.paInt16
RATE = 16000
RECORD_SECONDS = 5

def rms(data):
    'Function for calculating the Root Mean Square of sound data recorded through microphone device'   

    rms_value = 0
    n = len(data)    
    
    for sample_value in data:
        rms_value+=pow(sample_value,2)
    
    rms_value = np.sqrt(rms_value/n)

    return rms_value

def calibration_callback(keyword):
    capture_audio_flag = rospy.get_param('/unr_deepspeech/record_flag')
    
    if keyword.data=="calibrate" and capture_audio_flag==False:

        rospy.set_param('/unr_deepspeech/calibration_flag', param_value=True)

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

        recording_loudness = []
    
        print("* recording")        
        start_time = time.time()

        for i in range(0, int(RATE / CHUNK_SIZE * RECORD_SECONDS)):
            data = array('h', stream.read(CHUNK_SIZE, exception_on_overflow=False))
            recording_loudness.append(rms(data))
        
        print("* done recording")
        print("Time recording: "+str(time.time()-start_time))
        
        room_noise_threshold = mean(recording_loudness) + mean(recording_loudness)/2 # adding a small loudness portion (extra)
        
        print("Room Ambient Noise Threshold: "+str(room_noise_threshold))

        rospy.set_param('/unr_deepspeech/room_noise_threshold', room_noise_threshold)
        rospy.set_param('/unr_deepspeech/calibration_flag', param_value=False)

        stream.stop_stream()
        stream.close()
        p.terminate()


def calibration():
    rospy.init_node("calibration_node")
    calibration_sub = rospy.Subscriber("kws_data", String, calibration_callback)
    rospy.spin()

if __name__ == "__main__":
    calibration()