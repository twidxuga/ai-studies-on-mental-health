#!/bin/python 

#################################################
#
# Zero shot depression classification
#
# Data-set from: https://arxiv.org/abs/2202.03047
# Data-set reviewed/recommended by: https://link.springer.com/article/10.1007/s11831-022-09863-z
# Data-set Git Hub: https://github.com/Kayal-Sampath/detecting-signs-of-depression-from-social-media-postings
#
# Bart implementation information at: https://huggingface.co/facebook/bart-large-mnli
# Bart paper: https://arxiv.org/abs/1909.00161  
# 
#################################################

from tensorflow.python.ops.batch_ops import batch
from transformers import pipeline
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import torch
import numpy as np
import dtale

# Load the pipeline
pipe = pipeline(model="facebook/bart-large-mnli", device=0)

# sanity check 

pipe("I have a problem with my iphone that needs to be resolved asap!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)

testd = pd.read_csv(
    "./detecting-signs-of-depression-from-social-media-postings/test.tsv",
    sep="\t",
)
testd.columns = ['pid','text','label'] 
# 3245 data points

devd = pd.read_csv(
    "./detecting-signs-of-depression-from-social-media-postings/dev.tsv",
    sep="\t",
)
devd.columns = ['pid','text','label'] 
# 8891 data points

traind = pd.read_csv(
    "./detecting-signs-of-depression-from-social-media-postings/train.tsv",
    sep="\t",
)
traind.columns = ['pid','text','label'] 
# 4496 data point

# label values
out_labels = ['severe','moderate','not depression']


# rename columns for sanity and ease of use

testd.describe

dtale.show(testd)

testd.describe()
               # pid                                               text     label
# count         3245                                               3245      3245
# unique        3245                                               3233         3
# top     test_pid_1  Relaxing Saxophone Music for Stress Relief and...  moderate
# freq             1                                                  3      2169

testd.info()

testd.label.value_counts()
# moderate          2169
# not depression     848
# severe             228

testd.text.apply(lambda x : len(x)).describe()
# count     3245.000000
# mean       855.186441
# std       1061.059448
# min         13.000000
# 25%        244.000000
# 50%        548.000000
# 75%       1081.000000
# max      17342.000000

# MAXRECORDS = 100 
MAXRECORDS = len(testd) 

candidate_labels=[
    "severe depression",
    "moderate depression",
    "no depression",
    # "no depression",
    # "stress",
    # "anxiety",
    # "suicidal tendencies",
    # "anger",
    # "violence",
    # "sadness",
    # "happiness",
    # "neutral mood",
]

# run inference

%%time 
classified = pipe(
    list(testd.text)[:MAXRECORDS],
    batch_size = 10,
    candidate_labels=candidate_labels,
)

# test set runtime
# CPU times: user 22min 56s, sys: 42 s, total: 23min 38s
# Wall time: 22min 27s

# clear cache after every run to prevent running out of GPU memory
torch.cuda.empty_cache() 

# check results
results = pd.DataFrame(classified)
# results['isdepr'] = results.scores.apply(lambda x : x[0] > 0.5)
results['label'] = list(testd.label)[:MAXRECORDS] 

def class_deref(x):
    if x[0] == candidate_labels[0]:
        return out_labels[0]
    if x[0] == candidate_labels[1]:
        return out_labels[1]
    if x[0] == candidate_labels[2]:
        return out_labels[2]

results['rclass'] = results.labels.apply(class_deref)
results['match'] = results.label == results.rclass

# Overall performance (This will be bad)
metrics = precision_recall_fscore_support(
    results.label,
    results.rclass,
    labels = out_labels 
)

def print_metrics(metrics, labels):
    tpad = len(max(out_labels, key=len))
    print("\nLABEL" + " " * (tpad-5) + "\tPREC\tREC\tF1\tSUPP")
    for i, l in enumerate(labels):
        pad = tpad - len(l)
        print(f"{l}" + " " * pad \
          + f"\t{metrics[0][i]:.3f}\t{metrics[1][i]:.3f}\t{metrics[2][i]:.3f}\t{metrics[3][i]}")
    
print_metrics(metrics, out_labels)

# LABEL           PREC    REC     F1      SUPP
# severe          0.115   0.715   0.198   228
# moderate        0.629   0.491   0.551   2169
# not depression  0.373   0.059   0.102   848







# Attempt 2 - consolidate severe and moderate in the same depressed category

# consolidate severe and moderate from non-depression
testd['clabel'] = testd.label.apply(lambda x : x in [out_labels[0], out_labels[1]])

candidate_labels=[ "depression" ]

%%time
classified2 = pipe(
    list(testd.text)[:MAXRECORDS],
    batch_size = 10,
    candidate_labels=candidate_labels,
)

# CPU times: user 9min 7s, sys: 24.9 s, total: 9min 32s
# Wall time: 9min 10s

results = pd.DataFrame(classified2)
results['isdepr'] = results.scores.apply(lambda x : x[0] >= 0.5)
results['label'] = list(testd.clabel)[:MAXRECORDS] 

# Overall performance (This will be bad)
metrics = precision_recall_fscore_support(
    results.label,
    results.isdepr,
    labels = [True,False] 
)

print_metrics(metrics, ["True","False"])

# LABEL           PREC    REC     F1      SUPP
# True            0.815   0.612   0.699   2397
# False           0.356   0.607   0.449   848







# Attempt 3 - improve the context given to the model as class labels
# Also use consolidated severe and moderate annotations

testd['clabel'] = testd.label.apply(lambda x : x in [out_labels[0], out_labels[1]])

# using the definition of depression from NHS https://www.nhs.uk/mental-health/conditions/clinical-depression/overview/#overview

# candidate_labels=[ "I am depressed. I have depression. I have a long lasting low mood that affects my daily life. I'm feeling unhappy. I'm feeling hopeless. I have low self-esteem. I find no pleasure in things I usually enjoy." ]

# LABELS          PREC    REC     F1      SUPP
# True            0.739   0.997   0.849   2397
# False           0.222   0.002   0.005   848
# AVERAGES
# micro           0.737   0.737   0.737
# macro           0.480   0.500   0.427
# weighted        0.604   0.737   0.628


# candidate_labels=[ "depression. long lasting low mood that affects daily life." ]

# LABELS          PREC    REC     F1      SUPP
# True            0.793   0.802   0.797   2397
# False           0.421   0.407   0.414   848
# AVERAGES
# micro           0.699   0.699   0.699
# macro           0.607   0.604   0.605
# weighted        0.695   0.699   0.697


# candidate_labels=[ "depression. long lasting low mood. hopelessness. low self esteem" ]

# LABELS          PREC    REC     F1      SUPP
# True            0.742   0.986   0.847   2397
# False           0.441   0.031   0.057   848
# AVERAGES
# micro           0.737   0.737   0.737
# macro           0.591   0.508   0.452
# weighted        0.663   0.737   0.641


# candidate_labels=[ "depressed hopeless suicide" ]

# LABELS          PREC    REC     F1      SUPP
# True            0.869   0.328   0.477   2397
# False           0.312   0.860   0.457   848
# AVERAGES
# micro           0.467   0.467   0.467
# macro           0.590   0.594   0.467
# weighted        0.723   0.467   0.472


# candidate_labels=[ "depression" ]

# LABELS          PREC    REC     F1      SUPP
# True            0.815   0.612   0.699   2397
# False           0.356   0.607   0.449   848
# AVERAGES
# micro           0.611   0.611   0.611
# macro           0.586   0.610   0.574
# weighted        0.695   0.611   0.634


%%time
classified3 = pipe(
    list(testd.text)[:MAXRECORDS],
    batch_size = 10,
    candidate_labels=candidate_labels,
)

# CPU times: user 9min 24s, sys: 23.2 s, total: 9min 47s
# Wall time: 9min 24s

results = pd.DataFrame(classified3)
results['isdepr'] = results.scores.apply(lambda x : x[0] >= 0.5)
results['label'] = list(testd.clabel)[:MAXRECORDS] 

# better function to get metrics including 
def get_classification_metrics(y_true, y_pred, labels):
    tpad = len(max(out_labels + ['micro','macro','weighted','samples'], key=len))
    # report metrics per class
    metrics = precision_recall_fscore_support(y_true, y_pred, labels = labels)
    print("\nLABELS" + " " * (tpad-5) + "\tPREC\tREC\tF1\tSUPP")
    for i, l in enumerate(labels):
        print(f"{l}" + " " * (tpad - len(str(l))) + \
          f"\t{metrics[0][i]:.3f}\t{metrics[1][i]:.3f}\t{metrics[2][i]:.3f}\t{metrics[3][i]}")
    # check supported types at  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    print("AVERAGES")
    avg_types = ['micro','macro','weighted'] #,'samples']
    for avg in avg_types:
        metrics = precision_recall_fscore_support(y_true, y_pred, labels = labels, average = avg)
        print(avg + " " * (tpad - len(avg)) + \
              f"\t{metrics[0]:.3f}\t{metrics[1]:.3f}\t{metrics[2]:.3f}" ) 


get_classification_metrics(
    results.label,
    results.isdepr,
    labels = [True, False]
)


# baselines

# Compare with constant classifier 
get_classification_metrics(
    results.label,
    [True for i in results.label],
    labels = [True, False]
)

# LABELS          PREC    REC     F1      SUPP
# True            0.739   1.000   0.850   2397
# False           0.000   0.000   0.000   848
# AVERAGES
# micro           0.739   0.739   0.739
# macro           0.369   0.500   0.425
# weighted        0.546   0.739   0.628

# Compare with random classifier (with uniform distribution)
get_classification_metrics(
    results.label,
    np.random.rand(len(results.label)) > 0.5,
    labels = [True, False]
)

# LABELS          PREC    REC     F1      SUPP
# True            0.746   0.513   0.608   2397
# False           0.269   0.507   0.352   848
# AVERAGES
# micro           0.511   0.511   0.511
# macro           0.508   0.510   0.480
# weighted        0.622   0.511   0.541



# Attempt 4. Use a balanced dataset, extracted with random sampling

fulld = pd.concat([testd,devd,traind])
# [16632 rows x 3 columns]

fulld.label.value_counts()
# moderate          10479
# not depression     4663
# severe             1490
# Name: label, dtype: int64

# add column to reflect depression (or not)
fulld['clabel'] = fulld.label.apply(lambda x : x in [out_labels[0], out_labels[1]])
fulld.clabel.value_counts()
# True     11969
# False     4663
# Name: clabel, dtype: int64

NSAMPLES = 1500
trued = fulld[fulld.clabel == True].sample(NSAMPLES, random_state=42)
falsed = fulld[fulld.clabel == False].sample(NSAMPLES, random_state=42)
balanced = pd.concat([trued, falsed])


# baselines

# Compare with constant classifier 
get_classification_metrics(
    balanced.clabel,
    [True for i in results.label],
    labels = [True, False]
)

# LABELS          PREC    REC     F1      SUPP
# True            0.500   1.000   0.667   1500
# False           0.000   0.000   0.000   1500
# AVERAGES
# micro           0.500   0.500   0.500
# macro           0.250   0.500   0.333
# weighted        0.250   0.500   0.333


# Compare with random classifier (with uniform distribution)
get_classification_metrics(
    balanced.clabel,
    np.random.rand(len(results.label)) > 0.5,
    labels = [True, False]
)

# LABELS          PREC    REC     F1      SUPP
# True            0.499   0.517   0.508   1500
# False           0.499   0.481   0.490   1500
# AVERAGES
# micro           0.499   0.499   0.499
# macro           0.499   0.499   0.499
# weighted        0.499   0.499   0.499

# candidates

# candidate_labels = ["depression"]

# LABELS          PREC    REC     F1      SUPP
# True            0.620   0.600   0.610   1500
# False           0.613   0.633   0.622   1500
# AVERAGES
# micro           0.616   0.616   0.616
# macro           0.616   0.616   0.616
# weighted        0.616   0.616   0.616

candidate_labels=[ "depression, hopelessness, suicidal" ]

# LABELS          PREC    REC     F1      SUPP
# True            0.664   0.495   0.567   1500
# False           0.597   0.750   0.665   1500
# AVERAGES
# micro           0.622   0.622   0.622
# macro           0.631   0.622   0.616
# weighted        0.631   0.622   0.616



%%time
classified4 = pipe(
    list(balanced.text),
    batch_size = 10,
    candidate_labels=candidate_labels,
)


results = pd.DataFrame(classified4)
# results['isdepr'] = results.scores.apply(lambda x : x[0] >= 0.5)
# results['isdepr'] = results.scores.apply(lambda x : x[0] >= 0.383) # empirical cut-off 
results['isdepr'] = results.scores.apply(lambda x : x[0] >= 0.43987646) # linear regression 
results['label'] = list(balanced.clabel)

pos_scores = results[results.label == True].scores.apply(lambda x : x[0])
print(f"mean : {pos_scores.mean()}, std : {pos_scores.std()}")
neg_scores = results[results.label == False].scores.apply(lambda x : x[0])
print(f"mean : {neg_scores.mean()}, std : {neg_scores.std()}")

from sklearn.linear_model import LinearRegression 
clf = LinearRegression().fit(
    np.array(results.scores.apply(lambda x : x[0])).reshape(-1,1), 
    np.array(results.label)
)
clf.coef_

get_classification_metrics(
    results.label,
    results.isdepr,
    labels = [True, False]
)

# Playing with the cut-off parameter above 

# using 0.383 empirically obtained

# LABELS          PREC    REC     F1      SUPP
# True            0.655   0.586   0.619   1500
# False           0.626   0.692   0.657   1500
# AVERAGES
# micro           0.639   0.639   0.639
# macro           0.641   0.639   0.638
# weighted        0.641   0.639   0.638


# using 0.43987646 from linear regression

# LABELS          PREC    REC     F1      SUPP
# True            0.658   0.539   0.593   1500
# False           0.610   0.720   0.660   1500
# AVERAGES
# micro           0.630   0.630   0.630
# macro           0.634   0.630   0.627
# weighted        0.634   0.630   0.627

conclusion = '''

At this stage, and based only on the evaluation of this dataset, zero shot learning is not yet an effective option to detect clinical depression in social media posts.

This may be derived from the fact that the semantics of clinical depression, as evaluated by domain experts is different than the semantics or embeddings associated with current language depression related keywords.

I'll continue further exploration by using models trained with domain specific data-sets, and compare results in future. 

'''


