Machine Learning Toolkit in C++
=============

### There are 3 main components on MLT:

- **Models** : They perform the regression/classification/clustering task
- **Trainers** : They are used to train the models
- **Optimizers** : They minimize/maximize given functions

## Taxonomies of Models:
### Application:
- **Regressor** : Used to learn a regression task of a continuous value (or vector)
- **Label-Classifier** : Used to learn a classification task on a defined set of categories, they return only the label of one class
- **Support-Classifier** : Also used to learn a classification task, but they give the amount of support for each of the classes
- **Clusterer** : Used to group data on a defined number of clusters
- **Transformation** : Learns a transformation from the data (e.g. normalization) which can be latter applied to other data

### Parametrization:
- **Parametrized** : The model learns a fixed amount of parameters from the data, which later are used to calculate the prediction
- **Non-Parametrized** : The model keeps the data (or a transformation of it), and then uses it to calculate the prediction (this models usually are **Self-Trainable**)

### Method of training:
- **Self-Trainable** : Models that train themselves, without the need of an external trainers
- **Derivative-Free** : Models that provide a cost/fitness function, which can be used by derivative free optimization algorithms to evaluate the performance of a set of parameters
- **Gradient-Based** : Models that provide a cost/fitness function and its derivative, allowing for gradient based optimization algorithms to find the best parameters

Note that a model may fall into more than one of this categories, e.g. `LeastSquaresLinearRegressor` is **Self-Trainable** through the normal equations, and also **Grandient-Based** trained trough gradient descent (via the `cost` and `cost_gradients` functions); a `MultiLayerPerceptron` may give a vector of values with the support for each class, but the label of the class with the greatest support as well.

This taxonomies are not implemented as interfaces, to avoid the use of vtables, although on the future they may be implemented trough template metaprogramming, but it is guaranteed that methods of the same category have the same name (e.g. `self_train(...)` for self training, `regress(...)` for regression) for all the models (so you can exchange classes that fall within the same category)