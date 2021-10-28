
import os
from zipfile import ZipFile
bag_of_words = open('spans-pred_charbert.txt', 'r')
charBert = open('40_0.4_spans-pred.txt', 'r').readlines()
zipObj = ZipFile('spans-pred.zip', 'w')

def charList_to_intList(line):
	line = line.split('\t')
	line = line[1][1:-1].split(' ')
	span = []
	for elem in line:
		if len(elem) >= 2:
			span.append(int(elem[:-1]))
	return span

with open('spans-pred_.txt', 'w') as combined_preds:
	for i, line in enumerate(bag_of_words):
		spans = []
		span1 = charList_to_intList(line)
		span2 = charList_to_intList(charBert[i])
		for elem in span1:
			spans.append(elem)
		for elem in span2:
			spans.append(elem)
		spans = sorted(set(spans))
		combined_preds.write(str(i) + '\t' + str(spans) + '\n')

zipObj.write('spans-pred.txt')
# os.remove('spans-pred.txt')
