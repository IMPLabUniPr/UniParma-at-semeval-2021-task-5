import pandas as pd
from ast import literal_eval
from pprint import pprint
from nltk.tokenize import TweetTokenizer
import re

tTokenizer = TweetTokenizer()

def csv_to_df(data):
	train = pd.read_csv("tsd_" + data + ".csv")
	train["spans"] = train.spans.apply(literal_eval)
	return train

def separate_spans(df):
	sep_spans = {}
	for i, sentence in enumerate(df['text']):
		spans = []
		span = []
		for j, sp in enumerate(df['spans'][i]):
			if j == 0:
				span.append(sp)
			elif j == len(df['spans'][i])-1:
				span.append(sp)
				spans.append(span)
			else:
				if df['spans'][i][j+1] - df['spans'][i][j] == 1:
					span.append(sp)
				else:
					span.append(sp)
					spans.append(span)
					span = []
		sep_spans[str(i)] = spans
	return sep_spans


def return_label(word, tox_words):
	for tox_word in tox_words:
		tox_word = tTokenizer.tokenize(tox_word)
		if word in tox_word:
			if word == tox_word[0]:
				return 'B-LOC'
			else:
				return 'I-LOC'
		else:
			continue
	return 'O'

def extract_toxic_spans(sep_spans, df):
	txtfile = open('train.txt', 'w')
	for i, sentence in enumerate(df['text']):
		sentence_words = tTokenizer.tokenize(sentence)
		tox_words = []
		for span in sep_spans[str(i)]:
			tox = sentence[span[0]:span[-1]+1]
			tox_words.append(tox)

		tox_labels = {}
		for word in sentence_words:
			tox_labels[word] = return_label(word, tox_words)

		for key, value in tox_labels.items():
			line = key.split(' ')
			if len(line) >= 2:
				line = "".join(line)
				line = line + ' ' + value
				if len(line.split(' ')) == 2:
					txtfile.write(line + '\n')
			else:
				line = key + ' ' + value
				if len(line.split(' ')) == 2:
					txtfile.write(line + '\n')
		txtfile.write('\n')

def check_the_files():
	data = ['train', 'test']
	for datum in data:
		txtfile = open(datum + '.txt', 'r')
		clean = open(datum + '_ready.txt', 'w')
		for line in txtfile:
			if 'O' not in line and 'B-LOC' not in line and 'I-LOC' not in line and '.' in line:
				clean.write(line[:-1] + ' ' + 'O\n')
			else:
				clean.write(line)

def prepare_test():
	train = pd.read_csv("tsd_test.csv")
	with open('test.txt', 'w') as test:
		for i, row in enumerate(train['text']):
			words = tTokenizer.tokenize(row)
			print(words)
			for word in words:
				if '\n' in word:
					word = word.replace('\n', '.')
				word = word.split(' ')
				word = "".join(word)
				test.write(word + ' ' + 'O\n')
			test.write('\n')


def main():
	df = csv_to_df(data='train')
	sep_spans = separate_spans(df)
	extract_toxic_spans(sep_spans, df)
	prepare_test()
	check_the_files()


if __name__ == '__main__':
	main()