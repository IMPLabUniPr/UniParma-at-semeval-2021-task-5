from pprint import pprint
import unidecode
from unidecode import unidecode
import pandas as pd
from ast import literal_eval
from pprint import pprint
from nltk.tokenize import TweetTokenizer
import re
from collections import Counter
import json

def extract_statistics(file_name):
	toxic_words = []
	all_words = []
	stop_words = open('english_stop_words.txt', 'r')
	stop_words = [word[:-1] for word in stop_words]
	with open(file_name + '.txt', 'r') as file:
		for line in file:
			line = line.split(' ')
			if len(line) == 2:
				if line[1][:-1] == 'B-LOC' or line[1][:-1] == 'I-LOC':
					if line[0].lower() not in stop_words and line[0] not in ['.', ',', ':', '?', '/', ';', '-']:
						toxic_words.append(line[0].lower())
				all_words.append(line[0].lower())

	return Counter(toxic_words), Counter(all_words)

def find_beeped_words(spans, sentence):
	words = sentence.split(' ')
	for word in words:
		if '*' in word and word[0] != '*' and word[-1] != '*':
			index = sentence.find(word)
			if index >= 0:
				span = [j for j in range(index, index + len(word))]
			else:
				span = []
			spans.append(span)
	return spans

def make_predictions(stats, probs):
	test = pd.read_csv('tsd_test.csv')
	spans_pred = open('spans-pred_bow.txt', 'w')
	for i, sentence in enumerate(test.text):
		spans = []
		sentence = unidecode(sentence)
		spans = find_beeped_words(spans, sentence)
		for key in stats.keys():
			key = key.lower()
			index = sentence.lower().find(key)
			if index >= 0:
				if stats[key] >= 40 and probs[key] >= 0.7:
					span = [j for j in range(index, index + len(key))]
			else:
				span = []
			spans.append(span)
		clean_spans = []
		for item in spans:
			for elem in item:
				clean_spans.append(elem)
		spans =  sorted(set(clean_spans))
		spans_pred.write(str(i) + '\t' + str(spans) + '\n')

def write_stats_to_json(name, stats):
	st = {}
	with open(name + '.json', 'w') as stat:
		for key, value in stats.items():
			st[key] = value
		json.dump(st, stat)

def freq_to_prob(stats_toxic, stats_all):
	probs = {}
	count = 0
	for key, value in stats_toxic.items():
		prob = value/stats_all[key]
		if value >= 100:
			print(key, value, round(prob, 2))
		probs[key] = prob
	return probs

def main():
	stats_toxic, stats_all = extract_statistics('train')
	write_stats_to_json('toxic', stats_toxic)
	write_stats_to_json('all_words', stats_all)
	probs = freq_to_prob(stats_toxic, stats_all)
	predictions = make_predictions(stats_toxic, probs)
	import combine_two_predictions

if __name__ == '__main__':
	main()
