import re
import concurrent.futures

import pandas as pd

from os.path import join
from os import listdir, makedirs
from utilities import merge_workbooks, loadData, filterReview, clean_text


class DataProcessing():

    def balanceReviews(self, split_reviews, column_names):
        """
            Down-samples the reviews such that there are equal number of labels
            Paras:
                split_reviews : list of dataframes, where each dataframe is filtered by label
                column_names  : column names to keep for balanced dataframe 
            Returns:
                Balanced dataframe
        """
        length = min([len(split_reviews[i]) for i in range(0,5)])
        for i, reviews in enumerate(split_reviews):
            split_reviews[i] = split_reviews[i].sample(frac = 1)
            split_reviews[i] = reviews.iloc[:length, :]
        return merge_workbooks(split_reviews, column_names)

    def ProcessData(self):
        """
            Runs DataProcessing class and creates dataframe of cleaned reviews and associated rating labels
            
            Paras:
                None
            Returns:
                None
        """
        columns = ["summary", "reviewText", "overall"]
        df = loadData("./Dataset/raw_training_set", columns)

        split_reviews = [filterReview(df, "overall", i) for i in range(1,6)]
        df = self.balanceReviews(split_reviews, columns)
        df["reviews"] = df["summary"] + " " + df["reviewText"]

        reviews = df["reviews"].apply(lambda x: clean_text(x))
        ratings = df["overall"].tolist()
        
        return pd.DataFrame({"reviews": reviews.tolist(), "ratings": ratings})

if __name__ == "__main__":
    Preprocessing = DataProcessing()
    dataframe = Preprocessing.ProcessData()
