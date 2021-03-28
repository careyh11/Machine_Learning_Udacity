# '#'D OUT IMPORTS WERE USED AT SOME POINT IN PLAYING/PRODUCTION, CAH

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot
import tester
sys.path.append("../tools/")

#REMOVE PYTHON 2.7 WANRINGS, CAH
import warnings 
warnings.filterwarnings('ignore')

from time import time
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, cross_val_score
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from collections import defaultdict

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# FEATURES USED, CAH
label = 'poi'

fin_features = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',]

em_features = [
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',]

features_list = [label] + fin_features + em_features 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# NUMBER OF OBJECTS IN DATASET, CAH
print('There are {} objects in the dataset.'.format(len(data_dict)))

# POI VS. NON-POI, CAH
poi_counts = defaultdict(int)
for features_val in data_dict.values():
    poi_counts[features_val['poi']] += 1
print('There are {} POIs and {} non-POIs.'.format(poi_counts[True], poi_counts[False]))

### Task 2: Remove outliers

# REMOVE OUTLIERS
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
data_dict.pop('LOCKHART EUGENE E',0)

### Task 3: Create new feature(s)

# CREATE FEATURES, WITH MESSAGES TO_FROM POI AND ALL MESSAGES TO_FROM PERSON
# RETURN TO FRACTION MESSAGES TO_FROM PERSON THAT ARE TO_FROM POI
# 'fraction_from_poi', 'fraction_to_poi' ARE CREATED DURING THIS PROCESS, CAH

### Store to my_dataset for easy export below.
my_dataset = data_dict

def getFraction( poi_messages, all_messages ):
    
    fraction = 0.
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = float(poi_messages)/all_messages
    return fraction

for name in data_dict:

    var = data_dict[name]

    from_poi_to_this_person = var["from_poi_to_this_person"]
    to_messages = var["to_messages"]
    fraction_from_poi = getFraction(from_poi_to_this_person, to_messages)
    
    data_dict[name]["fraction_from_poi"] = fraction_from_poi
  
    from_this_person_to_poi = var["from_this_person_to_poi"]
    from_messages = var["from_messages"]
    fraction_to_poi = getFraction(from_this_person_to_poi, from_messages)

    data_dict[name]["fraction_to_poi"] = fraction_to_poi

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# SELECTKBEST, SHOW K TOP FEATS, CAH
selector = SelectKBest(k='all')
selectedFeatures = selector.fit(features, labels)
feature_names = [features_list[i+1] for i in selectedFeatures.get_support(indices=True)]
# scores = -np.log10(selector.pvalues_)
print('Best features in order: '), 
list(feature_names)#, (scores)

# SELECTKPERCENTILE, CUTTING FEATS IN HALF BY VALUE OF %, CAH
selector = SelectPercentile(f_classif, percentile = 50)
selectedFeatures = selector.fit(features, labels)
feature_names = [features_list[i+1] for i in selectedFeatures.get_support(indices=True)]
# scores = -np.log10(selector.pvalues_)
print('Best features in order: '), 
list(feature_names)#, (scores)

# FEATURES, CAH
# FEATURES CHOSEN BY USING THE TOP 4 FROM DEC TREE. I PLAYED WITH INPUT FROM GRIDSEARCH AND ADJUSTING FEATS
# BY HAND, RESET SOME VALUES FOR CHOSEN FEATS, CAH
label = 'poi'

# USING SPECIFIC TOP FEATURES FROM DECISION TREE, CAH
# THIS MAY BE HIGHLY BIAS'D, JUST A FEW FEATS, CAH
fin_features_final = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees']

features_list = [label] + fin_features_final

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# SPLIT DATA, TESTING AND TRAINING, CAH
features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                            labels, 
                                                                            shuffle = 100,
                                                                            test_size = 0.3, 
                                                                            random_state = 42)

# RUN DECISION TREE WITH CHOSEN FEATS AND TUNED SPECS FROM GRIDSEARCH
# THESE VALUES MAY NOT REPRPESENT WHAT IS ABOVE, BUT AFTER MANY RUNS THESE SEEM BEST, CAH
t0 = time()
clf = DecisionTreeClassifier(min_samples_split = 14,
                             min_samples_leaf = 1,
                             max_depth = 4,
                             max_features = 4)
clf.fit(features_train, labels_train).predict(features_test)
print ("training time:", round(time()-t0, 3), "s")
print ("-----------------------------------------")
print ("CLF Score:", clf.score(features_test, labels_test))
print ("-----------------------------------------")
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)