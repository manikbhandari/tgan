import numpy as np
import json, os, time, glob

from gan_dataiterator import *

class Data:
    def __init__(self, orig_sentence, orig_hidden_state, para_sentence, para_hidden_state):
        self.orig_sentence = orig_sentence
        self.orig_hidden_state = orig_hidden_state
        self.para_sentence = para_sentence
        self.para_hidden_state = para_hidden_state

class DataLoader:
    def __init__(self, source_file, file_type, metadata, subset_type="train"):
        if len(metadata) != 5:
            raise ValueError("Metadata cannot take values other than 5")

        data = self.load_file(source_file, file_type, metadata)
        self.dataset = DataSet(np.array(data.orig_sentence),
                          np.array(data.orig_hidden_state),
                          np.array(data.para_sentence),
                          np.array(data.para_hidden_state))

    def load_file(self, source_file, file_type, metadata):
        if file_type == "json":
            return self.load_json(source_file, metadata)
        else:
            raise NotImplementedError("file_type unrecognized")

    def load_json(self, source_file, metadata):
        all_orig_sentences = []
        all_orig_hidden_state = []
        all_para_sentences = []
        all_para_hidden_state = []

        with open(source_file, "r+") as f:
            data = json.load(f)

        for i, entry in enumerate(data[metadata[0]]):
            all_orig_sentences.append(entry[metadata[1].strip()])
            all_orig_hidden_state.append(entry[metadata[2].strip()])
            all_para_sentences.append(entry[metadata[3].strip()])
            all_para_hidden_state.append(entry[metadata[4].strip()])

        data = Data(all_orig_sentences, all_orig_hidden_state, all_para_sentences, all_para_hidden_state)
        return data
