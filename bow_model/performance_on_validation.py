from ast import literal_eval
import pandas as pd
import random
from evaluation.semeval2021 import f1
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 60})


tsd = pd.read_csv("data/tsd_val.csv")
tsd.spans = tsd.spans.apply(literal_eval)

probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0]

stats = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
         120, 130, 140, 150, 160, 170, 180, 190, 200]

plt.figure(figsize=(32, 18))
for i, prob in enumerate(probs):
    F1s = []
    for stat in stats:
        pred_name = 'all_val/' + str(int(stat/10)) + '_' + str(prob) + '_spans-pred.txt'
        tsd["random_predictions"] = pd.read_csv(pred_name, sep='\t', engine='python',
                                                            header=None, names=['Name'])
        tsd["f1_scores"] = tsd.apply(lambda row: f1(literal_eval(row.random_predictions), row.spans), axis=1)
        F1s.append(tsd.f1_scores.mean())
    plt.plot(stats, F1s, label=prob, linewidth=10)
plt.legend(title='Ratio', bbox_to_anchor=(1,1), loc="upper left")
# plt.legend(title='ratio', loc='upper right')
plt.xticks(stats, rotation ='vertical')
# plt.title('F1 score on validation set')
plt.ylabel('F1')
plt.xlabel('Frequency of toxic words in the resized training set')
plt.tight_layout()
plt.savefig('all_val', dpi=200)
