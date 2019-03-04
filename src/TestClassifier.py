import re
import pandas as pd

from os.path import join
from os import remove, system

from utilities import remove_columns, clean_text, save_data, make_dir, shell2var

class Test():

    def TestClassifier(self, name):
        """
            Constructs dataframe with test resutls

            Paras:
                None
            Return:
                None
        """

        # if make_dir("../Dataset/testing/train.csv"):

        df = pd.read_csv("../Dataset/test_set/train.csv")
        df = df.loc[df["categoryName"] == name]
        df = remove_columns(df, ["summary", "reviewText", "overall"])

        for _, temp in df.iterrows():
            data = temp.summary + ". " + temp.reviewText
            data = clean_text(data)
            data = temp.overall + " " + data + "\n"
            save_data("../Dataset/test_set/", "test_{}.txt".format(name), data, mode = "a")
        
        test_directory = join("../Dataset/test_set", "test_{}.txt".format(name))
        test_cmd = "../fastText/fasttext test ../fastTextModels/model_{}.bin {}".format(name, test_directory)
        return shell2var(test_cmd)

if __name__ == "__main__":
    name = "CDVinyl"
    test = Test()
    results = test.TestClassifier(name)
    print(results)