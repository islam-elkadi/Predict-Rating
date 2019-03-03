import re
import pandas as pd

from os import remove, system
from subprocess import Popen, PIPE

from utilities import Utils

class Test():

    def __init__(self):
        """
            Initalizes Test class with utilities
            Paras:
                None
            Returns:
                None
        """
        self.utls = Utils()

    def TestClassifier(self):
        """
            Constructs dataframe with test resutls

            Paras:
                None
            Return:
                None
        """

        df = self.utls.pd_csv("./Dataset/testing/train.csv")
        # df = df.loc[df["categoryName"] == "CDVinyl"]
        df = self.utls.remove_columns(df, ["summary", "reviewText", "overall"])

        for _, temp in df.iterrows():
            data = temp.summary + ". " + temp.reviewText
            data = self.utls.clean_text(data)
            data = temp.overall + " " + data + "\n"
            self.utls.save_data("./", "test_cdvyinyl.txt", data, mode = "a")
        
        system("./fastText/fasttext test ./fastTextModels/model_cdvinyl_4.bin test_cdvyinyl.txt")
        remove("test_cdvyinyl.txt")    

if __name__ == "__main__":
    classify = Test()
    classify.TestClassifier()