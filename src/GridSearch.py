from itertools import product
from utilities import make_dir, save_data

from TestClassifier import Test
from TrainClassifier import Train

class GridSearch(Test, Train):

    def __init__(self):
        super().__init__()

    def grid_search(self, kwargs):

        make_dir("../evaluations")
        wordNgrams = kwargs["wordNgrams"]
        bucket = kwargs["bucket"]
        lr = kwargs["lr"]
        dim = kwargs["dim"]
        epoch = kwargs["epoch"]
        loss = kwargs["loss"]

        args = product(wordNgrams, bucket, lr, dim, epoch, loss)
        
        for combinations in args:

            kwargs["wordNgrams"] = combinations[0]
            kwargs["bucket"] = int(combinations[1])
            kwargs["lr"] = combinations[2]
            kwargs["dim"] = combinations[3]
            kwargs["epoch"] = combinations[4]
            kwargs["loss"] = combinations[5]

            parameters = " ".join(map(str, [kwargs["wordNgrams"], kwargs["bucket"], kwargs["lr"], kwargs["dim"], kwargs["epoch"], kwargs["loss"]]))
            
            self.trainClassifier(**kwargs)
            results = "{}\n{}\n\n".format(parameters, self.testClassifier(kwargs["name"]))            
            save_data(directory = "../evaluations", name = "results.txt", docs = results, mode = "a")

if __name__ == "__main__":

    kwargs = {
        "name":"CDVinyl",
        "model":"supervised",
        "verbose":None,
        "minCount":None ,
        "minCountLabel":None,
        "wordNgrams":None,
        # "bucket":[2e6],
        "bucket":None,
        "minn":None,
        "maxn":None,
        "t":None,
        "label":None, 
        # "lr":[10e-2],
        "lr":None,
        # "loss":["ns", "softmax"],
        "loss":None,
        "lrUpdateRate":None,
        # "dim":[200, 300], 
        "dim": None,
        "ws":None,
        # "epoch":[10, 17], 
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