## Jupyter notebooks

In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.

If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.

And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

Be sure to put `%matplotlib inline` at the top of every code cell where you call plotting functions to get the resulting plots inside the document.

## Import the libraries

In Jupyter, select the cell below and press `ctrl + enter` to import the needed libraries.
Check out `labfuns.py` if you are interested in the details.

```
import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random
```

## Bayes classifier functions to implement

The lab descriptions state what each function should do.

```
# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
    # prior = (np.unique(labels, return_counts=True)[1]/Npts).reshape(-1, 1)
    # ==========================
    for jdx, c in enumerate(classes) :
        idx = np.where(labels == c)[0]
        wlc = W[idx,:]
        prior[jdx,:] = np.sum(wlc) / np.sum(W)

    # ==========================

    return prior

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    # for jdx, c in enumerate(classes):
    #     idx = np.where(labels == c)[0]
    #     xlc = X[idx,:]
    #     mu[jdx] = np.mean(xlc, axis=0)
    #     t = xlc - mu[jdx]
    #     sigma[jdx] = np.diag(np.diag(np.dot(t.transpose(), t)/xlc.shape[0]))
    # ==========================

    for jdx, c in enumerate(classes) :
        idx = np.where(labels == c)[0]
        xlc = X[idx,:]
        wlc = W[idx,:]
        mu[jdx] = np.dot(wlc.transpose(),xlc)/np.sum(wlc,axis=0)
        #sigma[jdx] = np.diag((np.dot(wlc.transpose(),pow(xlc-mu[jdx],2))/np.sum(wlc,axis=0)))
        sigma[jdx] = np.diag((np.dot(np.transpose(wlc), (xlc - mu[jdx,:])**2) / np.sum(wlc))[0])

    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================
    for i in range(0, Nclasses) :
        p1 = -0.5 * np.log(np.linalg.det(sigma[i]))
        inve = np.diag(1.0 / np.diag(sigma[i]))
        #p2 = -0.5 * np.diag(np.dot(np.dot((X - mu[i]), inve),np.transpose(X - mu[i])))
        p2 = -0.5 * np.diag(np.dot(np.dot((X - mu[i]), inve),np.transpose(X - mu[i])))
        p3 = np.log(prior[i])
        logProb[i,:] = p1 + p2 + p3
    # ==========================
    
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h
```

The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:

```
# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)
```

## Test the Maximum Likelihood estimates

Call `genBlobs` and `plotGaussian` to verify your estimates.

```
%matplotlib inline

X, labels = genBlobs(centers=5)
mu, sigma = mlParams(X,labels)
plotGaussian(X,labels,mu,sigma)
```

![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_3/assets/png/plotGaussian.png)

```
testClassifier(BayesClassifier(), dataset='iris', split=0.7)
```

Trial: 0 Accuracy 84.4  
Trial: 10 Accuracy 95.6  
Trial: 20 Accuracy 93.3  
Trial: 30 Accuracy 86.7  
Trial: 40 Accuracy 88.9  
Trial: 50 Accuracy 91.1  
Trial: 60 Accuracy 86.7  
Trial: 70 Accuracy 91.1  
Trial: 80 Accuracy 86.7  
Trial: 90 Accuracy 91.1  
Final mean classification accuracy  89 with standard deviation 4.16

```
plotBoundary(BayesClassifier(), dataset='iris',split=0.7)
```

![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_3/assets/png/BayesClassifierIris.png)

---

```
testClassifier(BayesClassifier(), dataset='vowel', split=0.7)
```

Trial: 0 Accuracy 61  
Trial: 10 Accuracy 66.2  
Trial: 20 Accuracy 74  
Trial: 30 Accuracy 66.9  
Trial: 40 Accuracy 59.7  
Trial: 50 Accuracy 64.3  
Trial: 60 Accuracy 66.9  
Trial: 70 Accuracy 63.6  
Trial: 80 Accuracy 62.3  
Trial: 90 Accuracy 70.8  
Final mean classification accuracy  64.7 with standard deviation 4.03

```
plotBoundary(BayesClassifier(), dataset='vowel',split=0.7)
```

![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_3/assets/png/BayesClassifierVowel.png)

## Boosting functions to implement

The lab descriptions state what each function should do.

```
# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        for i in range(0, Npts):
            if vote[i] == labels[i]:
                result[i] = 1
        theta = np.dot(wCur.transpose(), 1 - result)
        alpha = 0.5 * (np.log(1 - theta) - np.log(theta))
        alphas.append(alpha)
        for j in range(0, Npts):
            if vote[j] == labels[j]:
                newWeight[j] = np.exp(-alpha)
            else:
                newWeight[j] = np.exp(alpha)
        wCur = (wCur * newWeight) / np.sum(wCur * newWeight)
        # ==========================
        
    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        for i in range(0, Ncomps) :
            vote = classifiers[i].classify(X)
            for j in range(0, Npts) :
                votes[j,vote[j]] += alphas[i]
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)
```

```
The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.
```

```
# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)
```

```
testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)
```

Trial: 10 Accuracy 100  
Trial: 20 Accuracy 93.3  
Trial: 30 Accuracy 91.1  
Trial: 40 Accuracy 97.8  
Trial: 50 Accuracy 93.3  
Trial: 60 Accuracy 93.3  
Trial: 70 Accuracy 97.8  
Trial: 80 Accuracy 95.6  
Trial: 90 Accuracy 93.3  
Final mean classification accuracy  94.1 with standard deviation 6.72

```
plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)
```

![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_3/assets/png/BoostClassifierIris.png)

---

```
testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)
```

Trial: 0 Accuracy 76.6  
Trial: 10 Accuracy 86.4  
Trial: 20 Accuracy 83.1  
Trial: 30 Accuracy 80.5  
Trial: 40 Accuracy 72.7  
Trial: 50 Accuracy 76  
Trial: 60 Accuracy 81.8  
Trial: 70 Accuracy 82.5  
Trial: 80 Accuracy 79.9  
Trial: 90 Accuracy 83.1  
Final mean classification accuracy  80.2 with standard deviation 3.52

```
plotBoundary(BoostClassifier(BayesClassifier()), dataset='vowel',split=0.7)
```

![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_3/assets/png/BoostClassifierVowel.png)

Now repeat the steps with a decision tree classifier.

```
testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)
```

Trial: 0 Accuracy 95.6  
Trial: 10 Accuracy 100  
Trial: 20 Accuracy 91.1  
Trial: 30 Accuracy 91.1  
Trial: 40 Accuracy 93.3  
Trial: 50 Accuracy 91.1  
Trial: 60 Accuracy 88.9  
Trial: 70 Accuracy 88.9  
Trial: 80 Accuracy 93.3  
Trial: 90 Accuracy 88.9  
Final mean classification accuracy  92.4 with standard deviation 3.71

```
plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)
```

![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_3/assets/png/DecTreeClassifierIris.png)

---

```
testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)
```

Trial: 0 Accuracy 95.6  
Trial: 10 Accuracy 100  
Trial: 20 Accuracy 95.6  
Trial: 30 Accuracy 93.3  
Trial: 40 Accuracy 93.3  
Trial: 50 Accuracy 95.6  
Trial: 60 Accuracy 88.9  
Trial: 70 Accuracy 93.3  
Trial: 80 Accuracy 93.3  
Trial: 90 Accuracy 93.3  
Final mean classification accuracy  94.6 with standard deviation 3.65

```
plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)
```

![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_3/assets/png/BoostDecTreeClassifierIris.png)

---

```
testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)
```

Trial: 0 Accuracy 63.6  
Trial: 10 Accuracy 68.8  
Trial: 20 Accuracy 63.6  
Trial: 30 Accuracy 66.9  
Trial: 40 Accuracy 59.7  
Trial: 50 Accuracy 63  
Trial: 60 Accuracy 59.7  
Trial: 70 Accuracy 68.8  
Trial: 80 Accuracy 59.7  
Trial: 90 Accuracy 68.2  
Final mean classification accuracy  64.1 with standard deviation 4

```
plotBoundary(DecisionTreeClassifier(), dataset='vowel',split=0.7)
```

![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_3/assets/png/DecTreeClassifierVowel.png)

---

```
testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)
```

Trial: 0 Accuracy 84.4  
Trial: 10 Accuracy 89.6  
Trial: 20 Accuracy 86.4  
Trial: 30 Accuracy 93.5  
Trial: 40 Accuracy 84.4  
Trial: 50 Accuracy 79.9  
Trial: 60 Accuracy 89  
Trial: 70 Accuracy 86.4  
Trial: 80 Accuracy 85.7  
Trial: 90 Accuracy 85.7  
Final mean classification accuracy  86.6 with standard deviation 2.98

```
plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)
```

![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_3/assets/png/BoostDecTreeClassifierVowel.png)

# Answers to questions

> 1. When can a feature independence assumption be reasonable and when not?
> 2. How does the decision boundary look for the Iris dataset? How could one improve the classification results for this scenario by changing the classifier or, alternatively manipulating the data? 
> 3. Compute the classification accuracy of the boosted classifier on some data sets using `testclassifier` from `labfuns.py` and compare it with those of the basic classifier on the `iris` and `vowel` datasets.
>   > 1. Is there any improvement in classification accuracy? 
>   > 2. Plot the decision boundary of the boosted classifier on `iris` and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?
>   > 3. Can we make up for not using a more advanced model in the basic classifier (eg. independent features) by using boosting?
> 4. Same question as above, but for decision trees
>   > 1. Is there any improvement in classification accuracy? 
>   > 2. Plot the decision boundary of the boosted classifier on `iris` and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?
>   > 3. Can we make up for not using a more advanced model in the basic classifier (eg. independent features) by using boosting?
> 5. If you had to pick a classifier, naive Bayes or a decision tree or the boosted versions of these, which one would you pick? Motivate from the following criteria: 
>   > * Outliers
>   > * Irrelevant inputs: part of the feature space is irrelevant
>   > * Predictive power
>   > * Mixed types of data: binary, categorical or continous features, etc.
>   > * Scalability: the dimension of the data, D, is large or the number of instances, N, is large, or both. 

A naive bayes classifier assumes that the value of a particular feature is independent of the value of any other feature, given the class variable (conditional independence).

1. Is reasonable when several features are independent. For instance a persons weight, and how much hair he has. Is not not reasonable when there is no occurence between a class label and a feature, leading to the likelihood zero.

2. The decision boundary between class 1 and class 2 is quite incorrect. A lot of points belonging to class 1 are classified as class 2 instead. Instead of "bending to the left" it does the complete opposite.

Possible methods for improvement: boosting, using k-nn classifier? Change data: different points have different importance to decide boundary, give data points different weights so they can affect the boundary in different ways. Improve parameter change classifier: do not use simplified expressions (compute mu), or boosting.  

3. 
3.1 Classification for `iris` improved greatly, but `vowel` in general became worse graphically, but performed better, from 64% accuracy to 80%, while `iris` went from 89% to 94%. The decision boundary for `vowel` became simplified.  
3.2 Boundary became more complex, but the classification accuracy improved. Many class 2 points are now no longer classified as class 3.  
3.3 Use the weights from the last round of iteration and a basic Bayes classifier. 
4.   

4.1 Numerical accuracy improved (89% to 92%, 64% to 64%). 
4.2 Boundary became more complex. Combined weak classifiers form a strong classifier.
4.3 ???

5. Dec tree(high variance, low bias): easy to compute, can deal with irrelevant features, can deal with mixed data types, can calculate big training set, but easy to be overfitting. Easy to debug. Flexible. DecTree can handle both categorical (lab1)/binary and continous. Can handle mixed data because branching with discriminant features. Dectree tends to overfit... Payback by tuning tree (pruning) for performance. 

Bayes(high bias, low variance): fit situation of irrelvant between different dimensions and small training set, can deal with high-dimensional data but the result could be not good, sensitive to mixed types of data. Naive bayes require building classification by hand, can't just toss some data at it; need to pick features manually. Better for continous classification, not categorical. 

DecTerees work better (require) with large amounts of data, Bayes with more features and can tolerate lesser amounts of data. Naive baiyes used in robotics and computer vision. Pruning can sometimes be harmful in DecTrees. NB smaller and faster processing.

boosting: easy to be focused on error predictions too much, so it is sensitive to outliers.
