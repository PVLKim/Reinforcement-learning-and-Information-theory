# Possible template to start TP5

"""Compute the entropy of different models for text
            
Usage: compress [-m <model>] [-f <file>] [-o <order>]

Options:
-h --help      Show the description of the program
-f <file> --filename <file>  filename of the text to compress [default: Dostoevsky.txt]
-o <order> --order <order>  order of the model
-m <model> --model <model>  model for compression [default: IIDModel]
"""

import argparse, re
import numpy as np
import math
from collections import Counter


class IIDModel:
    """An interface for the text model"""
    def __init__(self, order=2):
        self.order = order
        self.src_dict = {}
        self.target_dict = {}
        pass
        
    def process(self, text, source=True):
        if source:
            for index in range(len(text)):
                symbol = text[index:index+self.order]
                if symbol not in self.src_dict:
                    self.src_dict[symbol] = 0
                self.src_dict[symbol] += 1
        else:
            for index in range(len(text)):
                symbol = text[index:index+self.order]
                if symbol not in self.target_dict:
                    self.target_dict[symbol] = 0
                self.target_dict[symbol] += 1
        pass

    def getEntropy(self):
        entropy = 0
        for key, value in self.src_dict.iteritems():
            p = value / float(sum(self.src_dict.values()))
            entropy += p * np.log2(p)
        return - entropy

    def getCrossEntropy(self, text):
        self.process(text, source = False)
        cross_entropy = 0
        src_sum = float(sum(self.src_dict.values()))
        target_sum = float(sum(self.target_dict.values()))
        for key, value in self.target_dict.iteritems():
            if key in self.src_dict.keys():
                p = self.src_dict[key] / src_sum
                q = self.target_dict[key] / target_sum
                cross_entropy += p * np.log2(q)
            else:
                q = self.target_dict[key] / target_sum
                cross_entropy += q * np.log2(q)
        for k, v in self.src_dict.iteritems():
            if k not in self.target_dict.keys():
                p = v / src_sum
                cross_entropy += p * np.log2(p)
        return -cross_entropy


class MarkovModel:
    """An interface for the text model"""
    def __init__(self, order=2):
        self.order = order
        self.src_seq = {}
        self.src_preced = {}
        self.target_seq = {}
        self.target_preced = {}
        pass

    def process(self, text, source=True):
        self.len_text = float(len(text))
        if source:
            temp_text = " " * self.order + text
            for index, char in enumerate(text):
                seq = temp_text[index : index + self.order + 1]
                if seq not in self.src_seq:
                    self.src_seq[seq] = 0
                self.src_seq[seq] += 1
                preced = temp_text[index : index + self.order]
                if preced not in self.src_preced:
                    self.src_preced[preced] = 0
                self.src_preced[preced] += 1
        else:
            temp_text = " " * self.order + text + " " * self.order
            for index, char in enumerate(text):
                seq = temp_text[index : index + self.order + 1]
                if seq not in self.target_seq:
                    self.target_seq[seq] = 0
                self.target_seq[seq] += 1
                preced = temp_text[index : index + self.order]
                if preced not in self.target_preced:
                    self.target_preced[preced] = 0
                self.target_preced[preced] += 1
        pass

    def getEntropy(self):
        entropy = 0
        for key, value in self.src_seq.iteritems():
            p_i = self.src_preced[key[:-1]] / self.len_text
            p = value / float(self.src_preced[key[:-1]])
            entropy += p_i * p * np.log2(p) / float(self.order)
        return - entropy

    def getCrossEntropy(self, text):
        self.process(text, source = False)
        self.len_target = float(len(text))
        self.unique_set = dict(Counter(text))

        cross_entropy = 0
        for key, value in self.target_seq.iteritems():
            q_i = self.target_preced[key[:-1]] / self.len_target
            q = value / float(self.target_preced[key[:-1]])
            if (key in self.src_seq.keys()) and (key[:-1] in self.src_preced.keys()):
                p = self.src_seq[key] / float(self.src_preced[key[:-1]])
                cross_entropy += q_i * p * np.log2(q) / float(self.order)
            else:
                cross_entropy += 1 /float(len(self.unique_set) * len(self.target_preced)) * np.log2(q) / float(self.order)
        return -cross_entropy


def preprocess(text):
    text = re.sub("\s\s+", " ", text)
    text = re.sub("\n", " ", text)
    return text

# Experiencing encoding issues due to UTF8 (on possibly other texts)? Consider:
#  f.read().decode('utf8')
#  blabla.join(u'dgfg')
#              ^

if __name__ == '__main__':
    from docopt import docopt
    # Retrieve the arguments from the command-line
    args = docopt(__doc__)
    print(args)

    # Read and preprocess the text
    src_text = preprocess(open(args["--filename"]).read())
    target_text = preprocess(open("Goethe.txt").read())

    # Create the model
    if(args["--model"]=="IIDModel"):
        model = IIDModel(int(args["--order"]))
    elif(args["--model"]=="MarkovModel"):
        model = MarkovModel(int(args["--order"]))

    model.process(src_text)
    print(model.getEntropy())
    print(model.getCrossEntropy(target_text))
    
