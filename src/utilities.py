import re

import pandas as pd

from os.path import exists, join
from subprocess import run, PIPE
from nltk.corpus import stopwords
from os import listdir, makedirs, system

#################
##  Pandas DF  ##
#################

def remove_columns(dataframe, column_names):
    """
        Filters out unwanted columns

        Paras:
            dataframe: dataframe to filter
            column_names: columns to keep

        Return:
            df: filtered dataframe
    """
    try:
        dataframe = dataframe[column_names]
        dataframe = dataframe.dropna()
        return dataframe.reset_index(drop = True)
    except (ValueError, KeyError):
        pass

def merge_workbooks(dataframes, column_names):
    """
        Merges multiple workbooks based on column names

        Paras:
            dataframes: dfs to merge
            Column_names: column names to merge on
        Returns:
            df: concatenated dataframes
    """
    return pd.concat([remove_columns(df, column_names) for df in dataframes])

def filterReview(df, name, value):
    """
        Filter dataframe by returning rows with column criteria
        Paras:
            df: dataframe
            name: name of column
            value: value to filter based on
        Return:
            df: data frame
    """
    return df.loc[df[name] == value]

#################
##  Load/Save  ##
#################

def loadData(path, column_names):
    """
        Loads the csvs in prepation for processing
        Paras:
            path: path
            column_names: columns to merge data on
        Return:
            df: merged dataframes
    """
    files = [doc for doc in listdir(path) if doc.endswith(".csv")]
    dataframes = [pd.read_csv(join(path, sheet)) for sheet in files]
    if len(dataframes)>1: return merge_workbooks(dataframes, column_names)
    else: return dataframes[0]

def make_dir(directory):
    """
        Checks if directory doesn't exist, then creates it.

        Paras:
            directory: name of directory

        Returns:
            Boolean
    """
    if not exists(directory): 
        makedirs(directory, exist_ok = True)
        return True
    else:
        return False

def save_data(directory, name, docs, mode = "w"):
    """
        Saves data to directory

        Paras:
            directory: directory to save data
            name: name of file
        Returns:
            None
    """
    make_dir(directory)
    with open(join(directory, name), mode, encoding = "utf-8") as f:
        f.write(docs)

def shell2var(cmd):
    """
        Runs shell command and returns output in a varible

        Paras:
            cmd: shell command
        Returns:
            output of shell command
    """
    result = run(args = cmd, stderr = PIPE, universal_newlines = True, stdout = PIPE, shell = True)   
    return result.stdout    

#################
##  Clean text ##
#################

def remove_contractions(raw):
    """
        Removes contractions to clean sentences
        
        Paras:
            raw: raw text data
        Returns:
            raw: cleaned text
    """
    contractions = { 
                    "ain't": "is not",
                    "aren't": "are not",
                    "can't": "cannot",
                    "could've": "could have",
                    "couldn't": "could not",
                    "didn't": "did not",
                    "doesn't": "does not",
                    "don't": "do not",
                    "hadn't": "had not",
                    "hasn't": "has not",
                    "haven't": "have not",
                    "he'd": "he would",
                    "he'll": "he will",
                    "he's": "he is",
                    "how'd": "how did",
                    "how'll": "how will",
                    "how's": "how is",
                    "I'd": "I would",
                    "I'll": "I will",
                    "I'm": "I am",
                    "I've": "I have",
                    "isn't": "is not",
                    "it'd": "it would",
                    "it'll": "it will",
                    "it's": "it is",
                    "let's": "let us",
                    "ma'am": "madam",
                    "mayn't": "may not",
                    "might've": "might have",
                    "mightn't": "might not",
                    "must've": "must have",
                    "mustn't": "must not",
                    "needn't": "need not",
                    "o'clock": "of the clock",
                    "oughtn't": "ought not",
                    "shan't": "shall not",
                    "sha'n't": "shall not",
                    "she'd": "she would",
                    "she'll": "she will",
                    "she's": "she is",
                    "should've": "should have",
                    "shouldn't": "should not",
                    "shouldn't've": "should not have",
                    "so've": "so have",
                    "so's": "so as",
                    "that'd": "that would",
                    "that's": "that is",
                    "there'd": "there had",
                    "there's": "there is",
                    "they'd": "they would",
                    "they'll": "they will",
                    "they're": "they are",
                    "they've": "they have",
                    "to've": "to have",
                    "wasn't": "was not",
                    "we'd": "we would",
                    "we'll": "we will",
                    "we're": "we are",
                    "we've": "we have",
                    "weren't": "were not",
                    "what'll": "what will",
                    "what're": "what are",
                    "what's": "what is",
                    "what've": "what have",
                    "when's": "when is",
                    "when've": "when have",
                    "where'd": "where did",
                    "where's": "where is",
                    "where've": "where have",
                    "who'll": "who will",
                    "who'll've": "who will have",
                    "who's": "who is",
                    "who've": "who have",
                    "why's": "why has",
                    "why've": "why have",
                    "will've": "will have",
                    "won't": "will not",
                    "won't've": "will not have",
                    "would've": "would have",
                    "wouldn't": "would not",
                    "y'all": "you all",
                    "you'd": "you had / you would",
                    "you'll": "you will",
                    "you'll've": "you will have",
                    "you're": "you are",
                    "you've": "you have"
                }
    if raw in contractions:
        return re.sub(raw, contractions[raw], raw)
    else:
        return raw

def clean_text(text):
    """
        Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings

        Paras:
            text: text data to clean
            remove_stopwords: if true, remove stop words from text to reduce noise
        Returns:
            text: cleaned text data
    """
    stops = set(stopwords.words("english"))
    text = text.encode("ascii", errors = "ignore").decode()
    text = [remove_contractions(word.lower()) for word in text.split()]
    text = " ".join([w for w in text if not w in stops])
    return  re.sub(r"[^a-zA-Z\s+\']", "", text)  
