# Answers to questions

## Assignment 0

monk1 training set has an even distribution between True/False, as can be seen in the entropy table below.

The third data set, with concepts (a5=1 A a4 = 1) V (a5 != 4 A a2 != 3) has very explicit rules, which should make "asking questions" easier. Unlike monk1 (a1 = a2) V (a5 = 1), which has the "fuzzy" condition (a1 = a2), which is not explicit in the value a1 or a2 is allowed to have, and thus is harder for the tree to learn. However monk1 does have the explicit condition (a5 = 1), which is easier than (ai = 1 for exactly two i in [1,6]) in monk2. 

By these arguments monk2 should be hardest to learn, and monk3 the easiest to learn. Of course some added noise in monk3 will make for a not completely accurate tree, but it will likely still be the most accurate tree.

## Assignment 1

Entropies

| Dataset   | Entropy                       |  
|-----------|-------------------------------|  
| Monk1     | 1.0                           |  
| Monk2     | 0.957117428264771 ~ 0.96      |
| Monk3     | 0.9998061328047111 ~ 0.99     |

## Assignment 2

Entropy can be explained as a measure of randomness, or "unorderlyness", or as defined in the lecture notes: measure of uncertainty (or unpredictability). Entropy for a uniform distributions is thus high, and in the previous assignment 1 for monk1, this is because the probability for getting any reading is the same as any other. This enthropy is higher because it is not more likely to get one measurement than another, so by this argument a gaussian distribution with a clear local maxima would have a lower enthropy, since probability is more concentrated around the expected value, ie it is more certain to find some value.

## Assignment 3

| Dataset | a1     | a2     | a3      | a4     | a5    | a6      |
|---------|--------|--------|---------|--------|-------|---------|
| Monk1   | 0.075  | 0.0058 | 0.0047  | 0.026  | 0.28  | 0.00075 |
| Monk2   | 0.0038 | 0.0025 | 0.0011  | 0.016  | 0.017 | 0.0062  |
| Monk3   | 0.0071 | 0.30   | 0.00083 | 0.0029 | 0.26  | 0.0071  |

By the above table, I'd sort monk1 after a5, monk2 by a5, and monk3 by a2. 

## Assignment 4

As described in the lab instructions, information gain is measured as: 

```gain = {enthropy before} - sum{normalized enthropy for every value k of attribute A}```

So when information gain is maximised, the second term needs to be as small as possible. This corresponds to the examples in the subset having higher certainty, ie more sorted data. 

Using information gain as a method of picking attribute of splitting in the end gives "orderly" and classified data. The reduction of enthropy shows that the classes after the split are more ordered, or after the split it is more certain to find one class than another in a leaf.

## Assignment 5

|       | Etrain | Etest  |
|-------|--------|--------|
| monk1 | 1.0    | 0.8287 |
| monk2 | 1.0    | 0.6921 |
| monk3 | 1.0    | 0.9444 |

Obviously the errors with using the training set is on point They were used to build the trees...

As assumed in assignment 0, monk3 test data generated the most accurate tree, while monk2 generated the most innaccurate tree.

## Assignment 6

High variance can cause the algorithm to model the noise in the training data (overfitting).  
High bias can make the algorithm miss relevant relations between features and target outputs. (underfitting) Bias measures difference between the model average and the "correct" value.

In decision trees, the depth of the tree determines the variance, ie the more nodes, the more difficult variations of the data there are, and thus higher variance, ie overfitting. Pruning the tree is then a way to reduce the variance.


## Assignment 7

See code for details.

monk1 validation data errors with dtree.check() :

![](https://gits-15.sys.kth.se/antolu/DD2421/raw/master/Lab_1/monk1mean.png)
![](https://gits-15.sys.kth.se/antolu/DD2421/raw/master/Lab_1/monk1mse.png)

monk3 validation data errors with dtree.check() :

![](https://gits-15.sys.kth.se/antolu/DD2421/raw/master/Lab_1/monk3mean.png)
![](https://gits-15.sys.kth.se/antolu/DD2421/raw/master/Lab_1/monk3mse.png)

We see that for both monk1 and monk3 that the validation performance increases as the fraction increases. However the MSE also increases linearly (at least it tends to do it linearly), this means the variance in validation data error also increases with the fraction, and one needs to find a fraction where both validation data performance and validation performance variance are optimal.