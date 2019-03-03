import json
import spacy

from utilities import makedir
from itertools import product
from collections import defaultdict

from Synonyms import Synonyms
from train_nns import WordRepresentation

class GridSearch(WordRepresentation):

    def __init__(self):
        super().__init__()

    def grid_search(self, kwargs, categories):
        evaluations = defaultdict(list)
        bucket = [2e6, 6e6, 10e6]
        lr = [10e-2, 10e-3, 10e-4]
        dim = [200, 300]
        epoch = [10, 17, 25]
        loss = ["ns", "hs"]
        args = product(bucket, lr, dim, epoch, loss)

        for i, combinations in enumerate(args):
            temp_evaluations = defaultdict(list)
            makedir("evaluations")
            kwargs["model"] = "cbow"
            kwargs["name"] = "{}_{}_{}".format("BookReviews", kwargs["model"], i)
            kwargs["wordNgrams"] = 6
            kwargs["bucket"] = int(combinations[0])
            kwargs["lr"] = combinations[1]
            kwargs["dim"] = combinations[2]
            kwargs["epoch"] = combinations[3]
            kwargs["loss"] = combinations[4]
            self.main(**kwargs)
            model = spacy.load("../SpaCy_models/{}".format(kwargs["name"]))
            syns = Synonyms(model).create_synonyms(categories)
            
            temp_evaluations["parameters"].append(combinations)
            temp_evaluations["results"].append(syns)

            evaluations[i].append(temp_evaluations)

        with open("evaluations/grid_search.json", mode = "w") as doc:
            json.dump(evaluations, doc, indent = 4, ensure_ascii = True)

if __name__ == "__main__":

    kwargs = {
        "name":None,
        "model":None,
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

    categories = ["theme", "plot", "character", "style", "dialogue"]

    gridsearch = GridSearch()
    gridsearch.grid_search(kwargs, categories)