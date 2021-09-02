import numpy as np
import os
import json, time, copy
import math

from tqdm import tqdm
import random
import pickle
from datetime import datetime
from pytz import timezone
from word2number import w2n
import string, re
from collections import Counter, defaultdict
from pprint import pprint
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner","textcat","parser"])
np.set_printoptions(precision=4)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Qcate_breakdown", type=str, default='["all"]')
parser.add_argument("--file", type=str)
parser.add_argument('--no_norm', action='store_true')
args = parser.parse_args()

import sys
sys.path.append("/home/yingshac/CYS/WebQnA/VLP/BARTScore")
from bart_score import BARTScorer

bart_scorer_ParaBank = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
bart_scorer_ParaBank.load(path='/home/yingshac/CYS/WebQnA/VLP/BARTScore/bart.pth') # Please change the path to bart.pth


def detectNum(l):
    result = []
    for w in l:
        try: result.append(str(int(w)))
        except: pass
    return result
def toNum(word):
    if word == 'point': return word
    try: return w2n.word_to_num(word)
    except:
        return word

def normalize_text(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text): # additional: converting numbers to digit form
        return " ".join([str(toNum(w)) for w in text.split()])

    def remove_punc(text):
        exclude = set(string.punctuation) - set(['.'])
        text1 = "".join(ch for ch in text if ch not in exclude)
        return re.sub(r"\.(?!\d)", "", text1) # remove '.' if it's not a decimal point

    def lower(text):
        return text.lower()
    
    def lemmatization(text):
        return " ".join([token.lemma_ for token in nlp(text)])

    if len(s.strip()) == 1:
        # accept article and punc if input is a single char
        return white_space_fix(lower(s))
    elif len(s.strip().split()) == 1: 
        # accept article if input is a single word
        return lemmatization(white_space_fix(remove_punc(lower(s))))

    return lemmatization(white_space_fix(remove_articles(remove_punc(lower(s)))))

# VQA Eval (SQuAD style EM, F1)
def compute_vqa_metrics(cands, a, exclude="", domain=None):
    if len(cands) == 0: return (0,0,0)
    bow_a = normalize_text(a).split()
    F1 = []
    EM = 0
    RE = []
    PR = []
    e = normalize_text(exclude).split()
    for c in cands:
        bow_c = [w for w in normalize_text(c).split() if not w in e]
        if domain == {"NUMBER"}: bow_c = detectNum(bow_c)
        elif domain is not None: 
            bow_c = list(domain.intersection(bow_c))
            bow_a = list(domain.intersection(bow_a))
        
        #print(bow_c)
        #print(bow_a)
        if bow_c == bow_a:
            EM = 1
        common = Counter(bow_a) & Counter(bow_c)
        num_same = sum(common.values())
        if num_same == 0:
            return (0,0,0,0,0)
        precision = 1.0 * num_same / len(bow_c)
        recall = 1.0 * num_same / len(bow_a)
        RE.append(recall)
        PR.append(precision)

        f1 = 2*precision*recall / (precision + recall + 1e-5)
        F1.append(f1)
    
    PR_avg = np.mean(PR)
    RE_avg = np.mean(RE)
    F1_avg = np.mean(F1)
    F1_max = np.max(F1)
    return (F1_avg, F1_max, EM, RE_avg, PR_avg)

color_set= {'orangebrown', 'spot', 'yellow', 'blue', 'rainbow', 'ivory', 'brown', 'gray', 'teal', 'bluewhite', 'orangepurple', 'black', 'white', 'gold', 'redorange', 'pink', 'blonde', 'tan', 'turquoise', 'grey', 'beige', 'golden', 'orange', 'bronze', 'maroon', 'purple', 'bluere', 'red', 'rust', 'violet', 'transparent', 'yes', 'silver', 'chrome', 'green', 'aqua'}
shape_set = {'globular', 'octogon', 'ring', 'hoop', 'octagon', 'concave', 'flat', 'wavy', 'shamrock', 'cross', 'cylinder', 'cylindrical', 'pentagon', 'point', 'pyramidal', 'crescent', 'rectangular', 'hook', 'tube', 'cone', 'bell', 'spiral', 'ball', 'convex', 'square', 'arch', 'h', 'cuboid', 'step', 'rectangle', 'dot', 'oval', 'circle', 'star', 'crosse', 'crest', 'octagonal', 'cube', 'triangle', 'semicircle', 'domeshape', 'obelisk', 'corkscrew', 'curve', 'circular', 'xs', 'slope', 'pyramid', 'round', 'bow', 'straight', 'triangular', 'heart', 'fork', 'teardrop', 'fold', 'curl', 'spherical', 'diamond', 'keyhole', 'conical', 'dome', 'sphere', 'bellshaped', 'rounded', 'hexagon', 'flower', 'globe', 'torus'}
yesno_set = {'yes', 'no'}


TABLE = str.maketrans(dict.fromkeys(string.punctuation)) 
def normalize_text_for_bart(x): # Light text normalization for WebQA eval: white space fix + punctuation removal
    return " ".join(x.translate(TABLE).split())

def compute_bartscore_ParaBank(c, a, switch=False):
    c_removepunc = [normalize_text_for_bart(x) for x in c]
    a_removepunc = [normalize_text_for_bart(x) for x in a]
    if switch: score = np.exp(bart_scorer_ParaBank.score(c_removepunc, a_removepunc))
    else: score = np.exp(bart_scorer_ParaBank.score(a_removepunc, c_removepunc))
    return score
        
Qcate_breakdown = json.loads(args.Qcate_breakdown)

print("Use categories: ", Qcate_breakdown)
print("Use normalization = ", not args.no_norm)

# Please change the path to your output folder
with open(os.path.join("/home/yingshac/CYS/WebQnA/VLP/vlp/light_output/", args.file), "r") as fp:
    lines = fp.readlines()
    header = lines[0].strip().split('\t')
    rows = lines[1:]
key = dict(zip(header, range(len(header))))
pprint(key)
F1_avg_scores = []
F1_max_scores = []
EM_scores = []
RE_scores = []
PR_scores= []
fluency_scores = []
acc_scores = []
mul_scores = []

output_Q = []
output_A = []
output_N = []
output_O = []
output_KA = []
output_G = []
output_QC = []
# Guid	Qcate	Q	A	Keywords_A	Output_conf	Output
for r in tqdm(rows):
    
    datum = r.strip().split('\t')
    Qcate = datum[key['Qcate']]
    if (not 'all' in Qcate_breakdown) and (not Qcate in Qcate_breakdown): continue
    O = json.loads(datum[key['Output']])
    C = [O[0]]
    Keywords_A = datum[key['Keywords_A']]
    A = json.loads(datum[key['A']])
    #normalizer = guid2norm[datum[key['Guid']]]
    normalizer = compute_bartscore_ParaBank(A, A)
    
    output_Q.append(datum[key['Q']])
    output_A.append(A)
    output_N.append(normalizer)
    output_O.append(O)
    output_KA.append(Keywords_A)
    output_G.append(datum[key['Guid']])
    output_QC.append(Qcate)
    
    if args.no_norm:
        score = min(1, np.max(compute_bartscore_ParaBank(C*len(A), A)))
    else: score = min(1, np.max(compute_bartscore_ParaBank(C*len(A), A)/np.array(normalizer)))

    if Qcate == 'color': F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(C, Keywords_A, "", color_set)
    elif Qcate == 'shape': F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(C, Keywords_A, "", shape_set)
    elif Qcate == 'YesNo': F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(C, Keywords_A, "", yesno_set)
    elif Qcate == 'number': F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(C, Keywords_A, "", {"NUMBER"})
    else: F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(C, Keywords_A)
    fluency_scores.append(score)
    if Qcate in ['color', 'shape', 'number', 'YesNo']: 
        acc_scores.append(F1_avg)
        mul_scores.append(F1_avg * score)
    else: 
        acc_scores.append(RE_avg)
        mul_scores.append(RE_avg * score)

    F1_avg_scores.append(F1_avg)
    F1_max_scores.append(F1_max)
    EM_scores.append(EM)
    RE_scores.append(RE_avg)
    PR_scores.append(PR_avg)

assert len(F1_avg_scores) == len(F1_max_scores) == len(EM_scores) == len(RE_scores) == len(PR_scores) == len(fluency_scores) == len(acc_scores) == len(mul_scores)
assert len(output_Q) == len(output_A) == len(output_N) == len(output_O) == len(output_KA) ==len(output_G) == len(output_QC) == len(mul_scores)
                    
print("#eval samples = ", len(mul_scores))
F1_avg = np.mean(F1_avg_scores)
F1_max = np.mean(F1_max_scores)
EM = np.mean(EM_scores)
RE_avg = np.mean(RE_scores)
PR_avg = np.mean(PR_scores)

fluency_avg = np.mean(fluency_scores)
acc_avg = np.mean(acc_scores)
mul_avg = np.mean(mul_scores)

print("F1_avg = {}".format(F1_avg))
#print("F1_max = {}".format(F1_max))
#print("EM = {}".format(EM))
print("RE_avg = {}".format(RE_avg))
#print("PR_avg = {}".format(PR_avg))
print("fluency_avg = {}".format(fluency_avg))
print("acc_avg = {}".format(acc_avg))
print("mul_avg = {}".format(mul_avg))
print(" ------------------------------------------------------------------------------------ \n")
if not 'all' in Qcate_breakdown: args.file = args.file.split(".")[0] + "_{}".format("|".join(Qcate_breakdown)) + ".tsv"

with open(os.path.join("/home/yingshac/CYS/WebQnA/VLP/vlp/light_output/", args.file.replace(".tsv", ".txt")), "w") as f:
    f.write(datetime.now(tz=timezone('US/Eastern')).strftime("%y-%m-%d %H:%M:%S") + '\n')
    f.write('\nUse Q categories: {}\nUse normalization = {}\n#Eval_samples = {}\n'.format(Qcate_breakdown, not args.no_norm, len(mul_scores)))
    f.write('\n --------------------- metrics -----------------------\n')
    f.write('\n'.join(["F1_avg = {}".format(F1_avg), "EM = {}".format(EM)]))
    f.write('\n\n')
    f.write('\n'.join(["RE_avg = {}".format(RE_avg), "PR_avg = {}".format(PR_avg)]))
    f.write('\n\n')
    f.write('\n'.join(["fluency_avg = {}".format(fluency_avg), "acc_avg = {}".format(acc_avg), "mul_avg = {}".format(mul_avg)]))
    f.write('\n\n')
    f.write('-----Starting writing results:-----')

    for guid, qcate, q, ka, a, n, o, re, f1, b, acc, m in zip(output_G, output_QC, output_Q, output_KA, output_A, output_N, output_O, RE_scores, F1_avg_scores, fluency_scores, acc_scores, mul_scores):
        f.write("\n\n")
        f.write("\n".join(['Guid === {}\n Qcate === {}'.format(guid, qcate), 
            'Q === {}'.format(q), 'Keywords_A === {}\n'.format(ka), '\n'.join(a), 
            'Normalizer === {}\n'.format(n), '\n'.join(o), 
            'RE === {}'.format(re), 'F1 === {}'.format(f1),
            'Fluency === {}'.format(b),
            'Accuracy === {}'.format(acc),
            "mul === {}".format(m),]))