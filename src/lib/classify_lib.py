import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score#classification_report

# Index of random permutation of data of length N
def randpermidx(N):
    return np.argsort(np.random.uniform(0, 1, N))

def linear_class_accuracy(data, labels):
    clf = LinearSVC(tol=1e-4, max_iter=2000)
    clf.fit(data, labels)
    return accuracy_score(labels, clf.predict(data))
    #return classification_report(labels, clf.predict(data), output_dict=True)['accuracy']

def linear_classifiability(data, labels, Nperm=100, print_freq=10):
    # Compute classification accuracy for true data-label pair
    rez = {
        'acc_true' : linear_class_accuracy(data, labels),
        'acc_shuffle' : []}
    
    # Shuffle data several times and recompute
    Ndata = len(labels)
    for i in range(Nperm):
        if print_freq is not None and i % print_freq == 0:
            print(i)
        labelsPerm = labels[randpermidx(Ndata)]
        rez['acc_shuffle'] += [linear_class_accuracy(data, labelsPerm)]
        
    return rez
    
    
# 1. Select all labeled data
# 2. Remove labeled data from pool
# 3. From remaining data select same number of points as labeled
# 4. Test classifier, permute, test again
# 5. Repeat multiple times
def linear_classifiability_eqpart(data, labels, Nperm=100, print_freq=100):
    Ndata = len(labels)
    NThis = int(np.sum(labels))
    NOther = Ndata-NThis
    maskThis = np.zeros(Ndata, dtype=bool)
    maskOther = np.ones(Ndata, dtype=bool)
    maskThis[labels==1] = 1
    maskOther[labels==1] = 0
    
    rez = {
        'acc_true' : [],
        'acc_shuffle' : []}
    
    labelsEq = np.hstack((np.ones(NThis), np.zeros(NThis)))
    for i in range(Nperm):
        if print_freq is not None and i % print_freq == 0:
            print(i)
            
        dataOtherEq = data[maskOther][randpermidx(NOther)][:NThis]
        dataEq = np.vstack((data[maskThis], dataOtherEq))
        
        rez['acc_true'] += [linear_class_accuracy(dataEq, labelsEq)]
        
        labelsShuffleEq = labelsEq[randpermidx(2*NThis)]
        rez['acc_shuffle'] += [linear_class_accuracy(dataEq, labelsShuffleEq)]
        
    return rez
    