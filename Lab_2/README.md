# 7 Exploring and reporting

> 1. Move the clusters around to make it easier or harder for the classifier to find a decent boundary. Pay attention to when the `minimize` function prints an error message that it can not find a solution.
> 2. Implement some of the non-linear kernels. you should be able to classify very hard datasets.
> 3. The non-linear kernels have parameters; explore how they influence the decision boundary. Reason about this in terms of the bias-variance trade-off.
> 4. Explore the role of the slack parameter C. What happens for very large/small values?
> 5. Imagine that you are given data that is not easily separable. When should you opt for more slack rather than going for a more complex model (kernel) and vice versa?

### 1. 

Using the provided datasets, a linear kernel provided a very tight-fitting separator. By adjusting the datasets as depicted in the second image below, renders a linear kernel non-functional. Instead a polynomial or radial kernel might be needed.

| Linear kernel with provided datasets | Linear kernel with nonlinear dataset |
| ------------------------------------ | ------------------------------------ |
| ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_linear_noslack.png "Linear kernel with provided datasets") | ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_linear_nonlineardata.png "Linear kernel with nonlinear datasets") |

### 2. 

Implementing the polynomial and radial kernel, the dataset that a linear kernel could not separate now became separable. 

The following plots without slack variables. 

| Polynomial kernel with provided datasets (p=2) | Polynomial kernel with provided datasets (p=3) |
| ------------------------------------ | ------------------------------------ |
| ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_polynomial_2_noslack.png "Polynomial kernel with provided datasets (p=2)") | ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_polynomial_3_noslack.png "Polynomial kernel with provided datasets (p=3)") |  

| Polynomial kernel with provided datasets (p=4) | Polynomial kernel with provided datasets (p=5) |
| ------------------------------------ | ------------------------------------ |
| ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_polynomial_4_noslack.png "Polynomial kernel with provided datasets (p=4)") | ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_polynomial_5_noslack.png "Polynomial kernel with provided datasets (p=5)") |  

| Radial kernel with provided datasets (sigma=1) | Radial kernel with provided datasets (sigma=2) |
| ------------------------------------ | ------------------------------------ |
| ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_radial_1_noslack.png "Radial kernel with provided datasets (sigma=1)") | ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_radial_2_noslack.png "Radial kernel with provided datasets (sigma=2)") |  

| Radial kernel with provided datasets (sigma=1) | Radial kernel with provided datasets (sigma=3) |
| ------------------------------------ | ------------------------------------ |
| ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_radial_3_noslack.png "Radial kernel with provided datasets (sigma=3)") | ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_radial_4_noslack.png "Radial kernel with provided datasets (sigma=4)") |  

#### Succcessful classification of nonlinearly separable dataset with polynomial and radial kernels 

| Polynomial kernel with nonlinear datasets (p=2) | Radial kernel with nonlinear datasets (sigma=2) |
| ------------------------------------ | ------------------------------------ |
| ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_polynomial_2_nonlineardata.png "Polynomial kernel with nonlinear datasets (p=2)") | ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_radial_2_nonlineardata.png "Radial kernel with nonlinear datasets (sigma=2)") |  

### 3. 

By increasing the degree of the polynomial in the polynomial kernel, we increase the bias of the model, ie we overfit the model to the training dataset. In the polynomial kernel the parameter p changes the degree of the polynomial while sigma in the radial kernel define the smoothness of the curve. 

By varying some of these parameters whe can make the model fit more closely or more loosely the dataset. If we increase the sigma of the radial kernel, for example, we obtain a smoother curve and increase the bias of the model while decreasing the variance. On the other hand if we decrease sigma we obtain a classification that tends to be more "edgy" and overfit the data (hence increase variance).

### 4. 

The standard SVM classifier works only if you have a well separated categories. To be more specific, they need to be linearly separable. It means there exist a line (or hyperplane) such that all points belonging to a single category are either below or above it. In many cases that condition is not satisfied, but still the two classes are pretty much separated except some small training data where the two categories overlap. It wouldn’t be a huge error if we would draw a line (somewhere in between) and accept some level of error - having training data on the wrong side of the marginal hyperplanes. How do we measure the error? The answer is: slack variables. For each training data point we can define a variable that measures the distance of the point to its marginal hyperplane, lets call it ξi. Whenever the point is on the wrong site of the marginal hyperplane we quantify the amount of error by the ratio between ξi and half of the margin, i.e. distance between separating hyperplane and marginal hyperplane (M in the figure). Points on the correct site are not quantified as errors. This is a geometrical interpretation of slack variables ξi. You can now go back to the initial SVM problem and maximize the margin in the presence of errors. The larger the error that you allow for, the wider the margin.

The larger C is, the larger the smaller the decision boundary margin is. Ie the larger the parameters alpha are allowed to be. 

#### Illustration of slack parameter C with linearly overlapping datasets

| Linear kernel with overlapping datasets (C=0.5) | Linear kernel with overlapping datasets (C=1) | Linear kernel with overlapping datasets (C=2)
| ------------------------------------ | ------------------------------------ | ------------------------------------ |
| ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_linear_slack_C05.png "Linear kernel with overlapping datasets (C=0.5)") | ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_linear_slack_C1.png "Linear kernel with overlapping datasets (C=1)") | ![](https://gits-15.sys.kth.se/antolu/DD2421/blob/master/Lab_2/assets/png/svmplot_linear_slack_C2.png "Linear kernel with overlapping datasets (C=2)") |

### 5. 

Tradeoff: maximize margin / minimize error. Increase value of C parameter: weight of missclassified points increase => margin gets smaller.   

A more complex kernel might overfit the training data, while the increased slack tolerance will likely provide better generilization. To this end, a more complex kernel will decrease bias and increase variance, while underfitting with a simpler model and higher slack might underfit the training set and yield high bias and low variance. 
