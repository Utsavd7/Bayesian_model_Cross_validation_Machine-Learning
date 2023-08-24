# Bayesian_model_Cross_validation_Machine-Learning

**1. Overview**

Naïve Bayes Classifier uses the Bayes’ theorem to predict membership probabilities for each class such as the probability that a given record or data point belongs to a particular class. The class with the highest probability is considered the most likely class. This is also known as the Maximum A Posteriori (MAP).

The MAP for a hypothesis with 2 events A and B is

MAP (A)
= max (P (A | B))

= max (P (B | A) * P (A))/P (B)

= max (P (B | A) * P (A))

Here, P (B) is evidence probability. It is used to normalize the result. It remains the same, So, removing it would not affect the result.


**2. Import libraries**

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
%matplotlib inline


**3. Check accuracy score**

Model accuracy score: 0.8057

Training set score: 0.8072
Test set score: 0.8057

So, there is no sign of overfitting.

Null accuracy score: 0.7582

We can see that our model accuracy score is 0.8083 but null accuracy score is 0.7582. So, we can conclude that our Gaussian Naive Bayes Classification model is doing a very good job in predicting the class labels.

Now, based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.

But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making.
