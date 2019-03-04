import re
import pandas as pd

from os.path import join
from nltk.tokenize import word_tokenize

from utilities import save_data
from DataProcess import DataProcessing

class DataSets(DataProcessing):

	def __init__(self):
		"""
			Inherits preprocessing class
			
			Paras:
				None
			Returns:
				None
		"""
		super().__init__()

	def createTrainingCorpus(self, df, name):
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
			save_data("../Dataset/training_processed", "{}.txt".format(name), doc + "\n", mode = "a")

	def createSet(self):
		"""
			Runs DataSets class and creates training set

			Paras: 
				None
			Returns:
				None
		"""	
		name = "CDVinyl"
		df = self.ProcessData()
		self.createTrainingCorpus(df, name)
		
if __name__ == "__main__":
	data = DataSets()
	data.createSet()