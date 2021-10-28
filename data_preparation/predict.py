
import pandas as pd
import json
from nltk.tokenize import TweetTokenizer
from unidecode import unidecode

tTokenizer = TweetTokenizer()

test = pd.read_csv('tsd_test.csv')
preds = json.load(open('predictions.json', 'r'))
test_data = json.load(open('data_test.json', 'r'))
spans_pred = open('spans-pred.txt', 'w')

for i, sentence in enumerate(test.text):
	spans = []
	items = []
	for j, item in enumerate(preds[str(i+1)]):
		if item != 'O':
			items.append((j, item))
			sentence = unidecode(sentence)
			index = sentence.lower().index(test_data[i][1][j])
			if j != len(preds[str(i+1)])-1:
				if preds[str(i+1)][j+1] != 'O':
					span = [i for i in range(index, index+len(test_data[i][1][j])+1)]
				else:
					span = [i for i in range(index, index+len(test_data[i][1][j]))]
			spans.append(span)
	print(items)
	clean_spans = []
	for item in spans:
		for elem in item:
			clean_spans.append(elem)
	spans =  sorted(set(clean_spans))
	spans_pred.write(str(i) + '\t' + str(spans) + '\n')

spans_pred.close()
