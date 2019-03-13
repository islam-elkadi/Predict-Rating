import re
import pandas as pd

from os.path import join, isfile
from os import remove, system

from utilities import remove_columns, clean_text, save_data, make_dir, shell2var

class Test():

    def createTestingCorpus(self, name):
        """
            Constructs dataframe with test resutls

            Paras:
                None
            Return:
                None
        """

        df = pd.read_csv("../Dataset/test_set/train.csv")
        df = df.loc[df["categoryName"] == name]
        df = remove_columns(df, ["summary", "reviewText", "overall"])
        df["reviews"] = df["summary"] + ". " + df["reviewText"]
        df["reviews"] = df["reviews"].apply(lambda x: clean_text(x))

        for _, temp in df.iterrows():
            data = temp.overall + " " + temp.reviews + "\n"
            save_data("../Dataset/test_set/", "test_{}.txt".format(name), data, mode = "a")

    def testClassifier(self, name):
        if not isfile("../Dataset/test_set/test_{}.txt".format(name)):
            print("IN HERE")
            self.createTestingCorpus(name)

        print("TESTING")
        test_directory = join("../Dataset/test_set", "test_{}.txt".format(name))
        print(test_directory)
        test_cmd = "../fastText/fasttext test ../fastTextModels/model_{}.bin {}".format(name, test_directory)
        print(test_cmd)
        return shell2var(test_cmd)

if __name__ == "__main__":
    name = "CDVinyl"
    test = Test()
    results = test.testClassifier(name)
    print(results)