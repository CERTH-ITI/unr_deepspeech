#!/usr/bin/env python

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

def get_room_threshold(pyaudio_format, chunk_size, sampling_rate, recording_time):

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio_format, channels=1, rate=sampling_rate, input=True, frames_per_buffer=chunk_size)

    recording_loudness_rms = []
    recording_loudness_max = []

    print("* Calibrating")        
    start_time = time.time()

    for i in range(0, int(sampling_rate / chunk_size * recording_time)):
        data = array('h', stream.read(chunk_size))
        recording_loudness_rms.append(rms(data))
        recording_loudness_max.append(max(data))

    print("* done Calibrating")

    room_noise_threshold_rms = 10*mean(recording_loudness_rms) # adding a small loudness portion (extra)
    room_noise_threshold_max = 5*mean(recording_loudness_max)

    stream.stop_stream()
    stream.close()
    p.terminate()

    return room_noise_threshold_rms, room_noise_threshold_max

def calibration_callback(keyword):
    record_flag = rospy.get_param('/unr_deepspeech/record_flag')
    
    if keyword.data=="calibrate" and record_flag ==False:

        rospy.set_param('/unr_deepspeech/calibration_flag', param_value=True)
        
        room_noise_threshold_rms, room_noise_threshold_max = get_room_threshold(FORMAT, CHUNK_SIZE, RATE, RECORD_SECONDS)
        
        print("Room Ambient Noise Threshold: "+str(room_noise_threshold_rms))

        rospy.set_param('/unr_deepspeech/room_noise_threshold_rms', room_noise_threshold_rms)
        rospy.set_param('/unr_deepspeech/room_noise_threshold_max', room_noise_threshold_max)
        rospy.set_param('/unr_deepspeech/calibration_flag', param_value=False)


def calibration():
    rospy.init_node("calibration_node")
    calibration_sub = rospy.Subscriber("kws_data", String, calibration_callback)
    rospy.spin()

if __name__ == "__main__":
    calibration()
