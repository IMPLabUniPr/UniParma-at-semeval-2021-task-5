

import pandas as pd
from ast import literal_eval
from pprint import pprint
from nltk.tokenize import TweetTokenizer
import re


tTokenizer = TweetTokenizer()


def read_csv_to_dataframe():
	trial = pd.read_csv("tsd_trial.csv")
	trial["spans"] = trial.spans.apply(literal_eval)
	return trial


def dataframe_to_txt(dataframe):
	trainfile = open('trial_train.txt', 'w')
	testfile = open('trial_test.txt', 'w')
	txtfile = trainfile
	for i, sentence in enumerate(dataframe['text']):
		words = tTokenizer.tokenize(sentence)
		for word in words:
			flag = False
			for span in dataframe['spans'][i]:
				spn = sentence[span[0]:span[-1]+1].split()
				if word in spn:
					flag = True
			if flag:
				txtfile.write(word + ' ' + 'B-LOC\n')
			else:
				txtfile.write(word + " " + "O\n")
		txtfile.write('\n')
		if i == 599:
			txtfile = testfile

def clean_data(data, name):
	wtrain = open('c' + name + '.txt', 'w')
	for line in data:
		words = line.strip().split(' ')
		if len(words)==1 and words[0].strip()!="":
			wtrain.write(line[:-1] + ' ' +'O\n')
			print(line[:-1])
		else:
			wtrain.write(line)

def main():
	dataframe = read_csv_to_dataframe()
	dataframe_to_txt(dataframe)
	train = open('trial_train.txt', 'r')
	clean_data(data=train, name='train')
	test = open('trial_test.txt', 'r')
	clean_data(data=test, name='test')

if __name__ == '__main__':
	main()