from deepspeech import Model
import sys
import wave
import struct


class DeepspeechNode:

    def __init__(self, model=None, model_path=None, scorer_path=None):
        self.model = model
        print("Model path:" + str(model_path))
        print("Scorer path:" + str(model_path))

        if model_path != None and scorer_path != None:
            self.load_model(model_path, scorer_path)

    def load_model(self, model_path,scorer_path):
        model_path = model_path + "/" + "deepspeech-0.8.2-models.pbmm"
        scorer_path = scorer_path +"/"+ "deepspeech-0.8.2-models.scorer"
        print("Model path: "+ model_path)
        print("Scorer path: " + scorer_path)
        
        self.model = Model(model_path)
        self.model.enableExternalScorer(scorer_path = scorer_path)

    def stt(self, fs, audio):
        assert self.model != None, "a model must be loaded before testing"
        transcription = self.model.stt(audio)

        return transcription
