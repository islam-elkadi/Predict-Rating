from os import system
from os.path import join
from utilities import make_dir, shell2var

class Train():

    def setParameters(self, **kwargs):

        """
            Parameters:
                The following arguments are mandatory:
                -input              training file path
                -output             output file path

                The following arguments are optional:
                -verbose            verbosity level [2]

                The following arguments for the dictionary are optional:
                -minCount           minimal number of word occurrences [5]
                -minCountLabel      minimal number of label occurrences [0]
                -wordNgrams         max length of word ngram [1]
                -bucket             number of buckets [2000000]
                -minn               min length of char ngram [3]
                -maxn               max length of char ngram [6]
                -t                  sampling threshold [0.0001]
                -label              labels prefix [__label__]

                The following arguments for training are optional:
                -lr                 learning rate [0.05]
                -lrUpdateRate       change the rate of updates for the learning rate [100]
                -dim                size of word vectors [100]
                -ws                 size of the context window [5]
                -epoch              number of epochs [5]
                -neg                number of negatives sampled [5]
                -loss               loss function {ns, hs, softmax} [ns]
                -thread             number of threads [12]
                -pretrainedVectors  pretrained word vectors for supervised learning []
                -saveOutput         whether output params should be saved [0]

                The following arguments for quantization are optional:
                -cutoff             number of words and ngrams to retain [0]
                -retrain            finetune embeddings if a cutoff is applied [0]
                -qnorm              quantizing the norm separately [0]
                -qout               quantizing the classifier [0]
                -dsub               size of each sub-vector [2]
            
            Returns:
                None
        """

        if kwargs["verbose"] == None: verbose = None
        else: verbose = "-verbose {}".format(kwargs["verbose"])
        
        if kwargs["minCount"] == None: minCount = None
        else: minCount = "-minCount {}".format(kwargs["minCount"])
        
        if kwargs["minCountLabel"] == None: minCountLabel = None
        else: minCountLabel = "-minCountLabel {}".format(kwargs["minCountLabel"])

        if kwargs["wordNgrams"] == None: wordNgrams = None
        else: wordNgrams = "-wordNgrams {}".format(kwargs["wordNgrams"])
        
        if kwargs["bucket"] == None: bucket = None
        else: bucket = "-bucket {}" .format(kwargs["bucket"])
        
        if kwargs["minn"] == None: minn = None
        else: minn = "-minn {}".format(kwargs["minn"])
        
        if kwargs["maxn"] == None: maxn = None
        else: maxn = "-maxn {}".format(kwargs["maxn"])

        if kwargs["t"] == None: t = None
        else: t = "-t {}".format(kwargs["t"])
        
        if kwargs["label"] == None: label = None
        else: label = "-label {}".format(kwargs["label"])

        if kwargs["lr"] == None: lr = None
        else: lr = "-lr {}".format(kwargs["lr"])
        
        if kwargs["lrUpdateRate"] == None: lrUpdateRate = None
        else: lrUpdateRate = "-lrUpdateRate {}".format(kwargs["lrUpdateRate"])
        
        if kwargs["dim"] == None: dim = None
        else: dim = "-dim {}".format(kwargs["dim"])
        
        if kwargs["ws"] == None: ws = None
        else: ws = "-ws {}".format(kwargs["ws"])
        
        if kwargs["epoch"] == None: epoch = None
        else: epoch = "-epoch {}".format(kwargs["epoch"])
        
        if kwargs["neg"] == None: neg = None
        else: neg = "-neg {}".format(kwargs["neg"])
        
        if kwargs["loss"] == None: loss = None
        else: loss = "-loss {}".format(kwargs["loss"])
        
        if kwargs["thread"] == None: thread = None
        else: thread = "-thread {}".format(kwargs["thread"])

        if kwargs["pretrainedVectors"] == None: pretrainedVectors = None
        else: pretrainedVectors = "-pretrainedVectors {}".format(kwargs["pretrainedVectors"])
        
        if kwargs["saveOutput"] == None: saveOutput = None
        else: saveOutput = "-saveOutput {}".format(kwargs["saveOutput"])

        if kwargs["cutoff"] == None: cutoff = None
        else: cutoff = "-cutoff {}".format(kwargs["cutoff"])

        if kwargs["retrain"] == None: retrain = None
        else: retrain = "-retrain {}".format(kwargs["retrain"])

        if kwargs["qnorm"] == None: qnorm = None
        else: qnorm = "-qnorm {}".format(kwargs["qnorm"])

        if kwargs["qout"] == None: qout = None
        else: qout = "-qout {}".format(kwargs["qout"])

        if kwargs["dsub"] == None: dsub = None
        else: dsub = "-dsub {}".format(kwargs["dsub"])
        
        cmd = [verbose, minCount, minCountLabel, wordNgrams, bucket, minn, maxn, t, label, lr, lrUpdateRate, dim, ws, epoch, neg, loss, thread, pretrainedVectors, saveOutput, cutoff, retrain, qnorm, qout, dsub]
        
        return " ".join(list(filter(None, cmd)))

    def trainClassifier(self, **kwargs):
        """
            Trains supervised classifier
            Paras:
                hyper_parameters: parameters to train neural net
            Returns:
                None
        """
        make_dir("../fastTextModels")
        name = kwargs["name"]
        model = kwargs["model"]
        parameters = self.setParameters(**kwargs)
        system("../fastText/fasttext {} -input ../Dataset/training_set_processed/training_{}.txt -output ../fastTextModels/model_{} -label __label__ {}".format(model, name, name, parameters))

class Test():

    def testClassifier(self, name):
        test_directory = join("../Dataset/test_set", "test_{}.txt".format(name))
        test_cmd = "../fastText/fasttext test ../fastTextModels/model_{}.bin {}".format(name, test_directory)
        return shell2var(test_cmd)

if __name__ == "__main__":
    kwargs = {
            "name": "CDVinyl",
            "model" :"supervised",
            "verbose": None,
            "minCount": None, 
            "minCountLabel": None, 
            "wordNgrams": None,
            "bucket": None,
            "minn": None,
            "maxn": None, 
            "t": None, 
            "label": None, 
            "lr": None, 
            "lrUpdateRate": None, 
            "dim": None, 
            "ws": None, 
            "epoch": None, 
            "neg": None, 
            "loss": None, 
            "thread": None,
            "pretrainedVectors": None, 
            "saveOutput": None, 
            "cutoff": None, 
            "retrain": None, 
            "qnorm": None, 
            "qout": None, 
            "dsub": None}
    train = Train()
    train.trainClassifier(**kwargs)
    test = Test()
    results = test.testClassifier(kwargs["name"])
    print(results)