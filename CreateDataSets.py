import re
import pandas as pd

from os.path import join
from nltk.tokenize import word_tokenize

from utilities import save_data
from DataProcess import DataProcessing

class DataSets(DataProcessing):

	def __init__(self):
		"""
			Initalizes DataSets class with utilities and preprocessing
			Paras:
				None
			Returns:
				None
		"""
		super().__init__()

	def createTrainingCorpus(self, df):
		"""
			Creates training data set with labels appended to beginging of each label

			Paras:
				df: datafframe
			Returns:
				None
		"""
		df = df.sample(frac = 1).reset_index(drop = True)
		ratings = df["ratings"].tolist()
		reviews = df["reviews"].tolist()

		for rating, review in zip(ratings, reviews):
			doc =  "__label__{}".format(rating) + " " + review.strip()
			doc = " ".join([word for word in word_tokenize(doc) if len(word)>1])
			save_data("./Dataset/training_processed", "training_amazon_video.txt", doc + "\n", mode = "a")

	def createSet(self):
		"""
			Runs DataSets class and creates training set

			Paras: 
				None
			Returns:
				None
		"""	
		df = self.ProcessData()
		self.createTrainingCorpus(df)
		
if __name__ == "__main__":
	data = DataSets()
	data.createSet()