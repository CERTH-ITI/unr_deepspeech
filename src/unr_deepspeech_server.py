#!/usr/bin/env python

from unr_deepspeech.srv import *
import rospy
from rospkg import RosPack

import wave
import struct

from deepspeech_node import DeepspeechNode

import sys
import glob
import os

def handle_listener(req):
    global listener

    audio_path = req.filename
    audio_file = wave.open(audio_path)
    fs = audio_file.getframerate()
    audio_string = audio_file.readframes(-1)
    audio = [struct.unpack("<h", audio_string[i:i+2])[0] for i in range(0, len(audio_string), 2)]

    text = listener.stt(fs, audio)
    print(text)

    return text

def listener_server():
    global listener

    rp = RosPack()
    package_path = rp.get_path("unr_deepspeech")
    model_path = "{}/model".format(package_path)

    if rospy.has_param("/unr_deepspeech/model"):
        model_path = rospy.get_param("/unr_deepspeech/model")

    print ("Loading model from : " + model_path)

    listener = DeepspeechNode(model_path=model_path)

    rospy.init_node('listener_server')

    rospy.Service('listen', Listen, handle_listener)
    print ("Ready to interpret speech audio.")

    rospy.spin()

if __name__ == "__main__":
    listener_server()
