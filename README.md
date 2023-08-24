# Bayesian_model_Cross_validation_Machine-Learning

**1. Overview**
Naïve Bayes Classifier uses the Bayes’ theorem to predict membership probabilities for each class such as the probability that a given record or data point belongs to a particular class. The class with the highest probability is considered the most likely class. This is also known as the Maximum A Posteriori (MAP).

The MAP for a hypothesis with 2 events A and B is

MAP (A)
= max (P (A | B))
= max (P (B | A) * P (A))/P (B)
= max (P (B | A) * P (A))

Here, P (B) is evidence probability. It is used to normalize the result. It remains the same, So, removing it would not affect the result.
