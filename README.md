# Bayesian_model_Cross_validation_Machine-Learning

_**1. Overview**_

Naïve Bayes Classifier uses the Bayes’ theorem to predict membership probabilities for each class such as the probability that a given record or data point belongs to a particular class. The class with the highest probability is considered the most likely class. This is also known as the Maximum A Posteriori (MAP).

The MAP for a hypothesis with 2 events A and B is

MAP (A)
= max (P (A | B))

= max (P (B | A) * P (A))/P (B)

= max (P (B | A) * P (A))

Here, P (B) is evidence probability. It is used to normalize the result. It remains the same, So, removing it would not affect the result.


_**2. Import libraries**_

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for data visualization purposes

import seaborn as sns # for statistical data visualization

%matplotlib inline


_**3. Check accuracy score**_

1) Model accuracy score: 0.8057
2) Training set score: 0.8072
3) Test set score: 0.8057

So, there is no sign of overfitting.

4) Null accuracy score: 0.7582

We can see that our model accuracy score is 0.8083 but null accuracy score is 0.7582. So, we can conclude that our Gaussian Naive Bayes Classification model is doing a very good job in predicting the class labels.

Now, based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.

But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifier is making.


_**4. Confusion matrix**_

Confusion matrix

 [[8992 2146]
 [ 701 2814]]

1) True Positives(TP) =  8992
2) True Negatives(TN) =  2814
3) False Positives(FP) =  2146
4) False Negatives(FN) =  701
   
The confusion matrix shows 5999 + 1897 = 7896 correct predictions and 1408 + 465 = 1873 incorrect predictions.

In this case, we have

1) True Positives (Actual Positive:1 and Predict Positive:1) - 5999
2) True Negatives (Actual Negative:0 and Predict Negative:0) - 1897
3) False Positives (Actual Negative:0 but Predict Positive:1) - 1408 (Type I error)
4) False Negatives (Actual Positive:1 but Predict Negative:0) - 465 (Type II error)

![download](https://github.com/Utsavd7/Bayesian_model_Cross_validation_Machine-Learning/assets/46219693/0d613c57-d8f4-4068-a2f7-8873c71aee3e)


_**5. Observations**_

![download](https://github.com/Utsavd7/Bayesian_model_Cross_validation_Machine-Learning/assets/46219693/6472d490-17bf-4aec-8dd8-03633f76be6c)

We can see that the above histogram is highly positively skewed.

The first column tells us that there are approximately 5700 observations with a probability between 0.0 and 0.1 whose salary is <=50K.

There are a relatively small number of observations with probability > 0.5.

So, these small number of observations predict that the salaries will be >50K.

The majority of observations predict that the salaries will be <=50K.

_**6. ROC Curve**_

![download](https://github.com/Utsavd7/Bayesian_model_Cross_validation_Machine-Learning/assets/46219693/03df7930-b0f5-44d3-bf6b-d6d9ead67fb2)


_**7. Results and conclusion**_

In this project, I build a Gaussian Naïve Bayes Classifier model to predict whether a person makes over 50K a year. The model yields a very good performance as indicated by the model accuracy which was found to be 0.8083.

The training-set accuracy score is 0.8067 while the test-set accuracy to be 0.8083. These two values are quite comparable. So, there is no sign of overfitting.

I have compared the model accuracy score which is 0.8083 with null accuracy score which is 0.7582. So, we can conclude that our Gaussian Naïve Bayes classifier model is doing a very good job in predicting the class labels.

ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a very good job in predicting whether a person makes over 50K a year.

Using the mean cross-validation, we can conclude that we expect the model to be around 80.63% accurate on average.

If we look at all the 10 scores produced by the 10-fold cross-validation, we can also conclude that there is a relatively small variance in the accuracy between folds, ranging from 81.35% accuracy to 79.64% accuracy. So, we can conclude that the model is independent of the particular folds used for training.

Our original model accuracy is 0.8083, but the mean cross-validation accuracy is 0.8063. So, the 10-fold cross-validation accuracy does not result in performance improvement for this model.




