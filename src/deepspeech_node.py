from deepspeech import Model
import sys
import wave
import struct


class DeepspeechNode:

    def __init__(self, model=None, model_path=None, scorer_path=None):
        self.model = model
        print("Model files path:" + str(model_path))

        if model_path != None:
            self.load_model(model_path)

    def load_model(self, model_path):
        init_model_path = model_path
         # output_graph.pbmm v0.6.1 accoustic model /// deepspeech-0.8.2-models.pbmm v0.8.2 accoustic model (latest)
        accoustic_model_path = init_model_path + "/" + "deepspeech-0.8.2-models.pbmm"
        language_model_path = init_model_path + "/" + "lm.binary"
        trie_path = init_model_path + "/" + "trie"

        print("Model path: " + accoustic_model_path)
        print("LM path: " + language_model_path)
        print("Trie path: " + trie_path)

        self.model = Model(accoustic_model_path, 250) #beam search width (the higher, the more time consuming the transcription)
        self.model.enableDecoderWithLM(aLMPath=language_model_path, aTriePath=trie_path, aLMAlpha=0.75, aLMBeta=1.85)

    def stt(self, fs, audio):
        assert self.model != None, "a model must be loaded before testing"
        transcription = self.model.stt(audio)

        return transcription
