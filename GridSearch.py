import json
import spacy

from itertools import product
from collections import defaultdict
from utilities import make_dir, save_data

from TestClassifier import Test
from TrainClassifier import Classify

class GridSearch(Test, Classify):

    def __init__(self):
        super().__init__()

    def grid_search(self, kwargs):
        make_dir("evaluations")
        bucket = [2e6, 6e6, 10e6]
        lr = [10e-2, 10e-3, 10e-4]
        dim = [200, 300]
        epoch = [10, 17, 25]
        loss = ["ns", "softmax"]
        args = product(bucket, lr, dim, epoch, loss)
        
        for i, combinations in enumerate(args):
            kwargs["name"] = "{}_{}_{}".format("AmazonInstantVideo", kwargs["model"], i)
            kwargs["wordNgrams"] = 6
            kwargs["bucket"] = int(combinations[0])
            kwargs["lr"] = combinations[1]
            kwargs["dim"] = combinations[2]
            kwargs["epoch"] = combinations[3]
            kwargs["loss"] = combinations[4]
            parameters = " ".join([kwargs["wordNgrams"], kwargs["bucket"], kwargs["lr"], kwargs["dim"], kwargs["epoch"], kwargs["loss"]])
            
            self.main_trainClassifier(**kwargs)
            results = "{}\n{}\n\n".format(parameters, self.TestClassifier(kwargs["name"]))            
            save_data(directory = "evaluations", name = "results.txt", docs = results, mode = "a")

if __name__ == "__main__":

    kwargs = {
        "name":None,
        "verbose":None,
        "minCount":None ,
        "minCountLabel":None,
        "wordNgrams":None,
        "bucket":None,
        "minn":None,
        "maxn":None,
        "t":None,
        "label":None, 
        "lr":None,
        "lrUpdateRate":None,
        "dim":None, 
        "ws":None,
        "epoch":None, 
        "neg":None,
        "thread":None,
        "pretrainedVectors":None,
        "saveOutput":None,
        "cutoff":None,
        "retrain":None,
        "qnorm":None,
        "qout":None,
        "dsub":None,
    }

    gridsearch = GridSearch()
    gridsearch.grid_search(kwargs)