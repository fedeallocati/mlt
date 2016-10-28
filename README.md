Machine Learning Toolkit in C++
=============

MLT is a header-only Machine Learning library.
It's main components are **Models**, which perform the regression/classification/clustering/transformation task at hand.

## Types of Models:
- **Regressors** : Used to learn a regression task of a continuous value (or vector of values)
- **Classifiers** : Used to learn a classification task on a defined set of categories, they return only the label of one class. Some also can return a score for each class.
- **Clusters** : Used to label and group unlabelled data
- **Transformers** : Learns a transformation from the data (e.g. normalization) which can be latter applied to other Model

## Conventions:
- Samples are expected to be passed as matrices with each sample as a column, that is: [n_features, n_samples]. The same goes for the target values in supervised learning: [n_output, n_samples], and also for the predictions of the Models.
