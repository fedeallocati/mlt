Machine Learning Toolkit in C++
=============

### The main component on MLT are **Models**, which perform the regression/classification/clustering/transformation task 

## Taxonomies of Models:
### Application:
- **Regressor** : Used to learn a regression task of a continuous value (or vector)
- **Label-Classifier** : Used to learn a classification task on a defined set of categories, they return only the label of one class
- **Support-Classifier** : Also used to learn a classification task, but they give the amount of support for each of the classes
- **Clusterer** : Used to group data on different clusters
- **Transformation** : Learns a transformation from the data (e.g. normalization) which can be latter applied to other data

### Parametrization:
- **Parametrized** : The model learns a fixed amount of parameters from the data, which later are used to calculate the prediction
- **Non-Parametrized** : The model keeps the data (or a transformation of it), and then uses it to calculate the prediction

### Supervision:
- **Supervised** : Models that take the expected output when training
- **Unsupervised** : Models that just take the input data when training

This taxonomies are not implemented as interfaces, to avoid the use of vtables, although on the future they may be implemented trough template metaprogramming, or CRTP, but it is guaranteed that methods of the same category have the same name (e.g. `fit(...)` for training, `predict(...)` for regression) for all the models (so you can exchange classes that fall within the same category and use them on templated code).

## Conventions:
- Samples are expected to be passed as matrices with each sample as a column, that is: [n_features, n_samples]. The same goes for the target values: [n_output, n_samples]