#Reading notes of: Hands-on machine learning with Scikit-learn, Keras & Tensorflow (Aurélien Géron)

##Chapter 1: The machine learning landscape.

#####General definitions of ML:

> The field of study that gives computers the ability to learn withour being explicilty programmed.  
> Arthur Samuel (1959)

> A computer program is said to learn from experience $E$ with respect to some task $T$ and some performance measure $P$, if its performance on $T$, as measured by $P$, improves with $E$.
> Tom Mitchell (1997)

#####Why using machine learning?
In constrast with hard-coding a set of rules defining a given behaviour, machine learning is devised to extract models from data permitting to mimic the behaviour. It can be used for:

- Simple problems requiring a lot of fine tuning and maintenance (updating frequently rules).
- Complex problems (or complex behaviours) that cannot be solved traditionally by an enumaration of rules. (e.g image)
- Dynamic structure: behaviours in the environment are evolving. The system needs to update the set of rules describing the behaviour.
- Knowledge extraction: gaining insight about an environement/subject by studying structures provided by ML algorithms. (Data Mining)

#####Types of machine learning

Classification of ML algorithms w.r.t 3 criteria: Supervision, Online/batch learning and, Instance-based/Model-based learning

######Supervision: What amount and type of supervision do an algorithm require ?

1. **Supervised algorithms**: The training data contain the desired outcom, called _labels_. Data can be represented as a function: $f(X) = y$ where $X$ is a vector of _attributes_ and $y$ the label. The problem is to learn $f$. Some supervised algorothms: k-nearest neighbors, regressions, SVM, decision trees, random forests, neural networks (not all of them)
2. **Unsupervised learning**: Training data do not contain labels. Data sets can be represented as a set of vectors $X$. The problem here is to find structure in the data ; e.g. are they in different groups ? is there correlations between attrbutes ? Some unsupervised tasks: clusturing (K-Means, DBSCAN, Hierachical cluster analysis), Anomaly and novelty detection, Visualisation and dimensionality reduction (Principal Component Analysis, Kernel PCA, locally linear embedding), association rule leaning (Apriori, Eclat)
3. **Semi-supervised learning**: dealling with partially labeled data. Algorithms in that category generally mixe supervised and unsupervised learning.
4. **Reinforcement learning**: The learning system do not really have training data but observes an environment, make actions (or predictions), gets rewarded or not and updates itself based on the reward or penalty. It learns by itself what is the best strategy by experience over time. The problem here is to actually be able to _converge_ and define the best strategy to do so quickly (updating function besed on _rewards/penalties_) (Note to self: read more about reinforcement learning and game theory)

######Batch and online learning: Can an algorithm upadate its model easily?
Very succintly, batch learning use availbale data in batches: algorithms learn a model once and do not update it. Updating models requires to re-run algorithms. Online learning algorithms update models with novel and incoming data incrementaly and sequentially using data nuggets or _mini-batches_.

######Instance-based vs. Model-based learning: how do algorithm _generalize_ ?

- Is generalization done by similarity ? Use of a similarity measure to compare new data to known data. If a given data point is similar w.r.t to a similarity measure (a distance in a given space), to known data points.
- Is generalization done by prediction given a model/representation ? New data points are processed and predictions are done using models (equations, graphs ...). (e.g regression)

#####Main challenges of machine learning

- Insuficient quantitiy of data
- Non representative training data. when data do not represent new cases to be generalized to (sampling biais)
- Poor-quality data: errors, outliers, amount of noise, missing attrbutes
- Irrelevant features (feature engineering)
- Overfitting training data: performing very good with training data but not being able to generalize (e.g patterns in noise). (regularization)
- Underfitting: model is too simple to effectively learn from data. _Pattern language's expressivity does not permit to capture a pattern complexity_.

#####Testing and validating
Spliting data sets into training set and test set. Common rule: use 80% of data for training (but depends on the training set size: if 10 million instances, 1% for testing is probably enough).

**Training error (TE)**: errors that a model makes with training set.
**Generalization error (GE)**: errors that a model makes with new instances.

If GE low and TE high: overfitting.

**Holdout validation**: Hold out a part of the training set (validation set, development set, dev set) to evaluate several models (i.e linear, logistic ...) with different hyperparameters. Chose the best model and train it on the entire training set (including the validation set). Then, evaluate _genealization_ on the test set.

**Data mismatch**: The validation and test sets must be as representative as possible data to expect in production. The objective is perform well for the expected data. Thus, you can train models on various data sets by always validate and test generalization on representative data sets.

**No free lunch theorem** (Copied from http://www.no-free-lunch.org/. accessed 24/10/2021)

Hume (1739–1740) pointed out that ‘even after the observation of the frequent or constant conjunction of objects, we have no reason to draw any inference concerning any object beyond those of which we have had experience’. More recently, and with increasing rigour, Mitchell (1980), Schaffer (1994) and Wolpert (1996) showed that bias-free learning is futile.

Wolpert (1996) shows that in a noise-free scenario where the loss function is the misclassification rate, if one is interested in off-training-set error, then there are no a priori distinctions between learning algorithms.

More formally, where

- $d$ = training set;
- $m$ = number of elements in training set;
- $f$ = ‘target’ input-output relationships;
- $h$ = hypothesis (the algorithm's guess for f made in response to d); and
- $C$ = off-training-set ‘loss’ associated with f and h (‘generalization error’)

all algorithms are equivalent, on average, by any of the following measures of risk: $E(C|d), E(C|m), E(C|f,d),$ or $E(C|f,m)$.

How well you do is determined by how ‘aligned’ your learning algorithm $P(h|d)$ is with the actual posterior, $P(f|d)$.

Wolpert's result, in essence, formalizes Hume, extends him and calls the whole of science into question.

##Chapter 2: End-to-end Machine Learning Project

Basic steps when dealing with a ML problem (notes correspond mostly to a regression problem):

- Look at the big picture
  - Decide wich type of ML problem (regression, classification) (supervised, unsupervised) (what type of data and what type of outcome)
  - Select a performance measure
  - Check assumptions
- Get the data
  - Always get insights (e.g historgrams) about data before doing anything (no free lunch theorem). Permits also to spots some elements to consider when preparing data for ML (spotting outliers, attributes unit ...) and chose an ML model. Permits also to spot some potential biaises.
  - When spliting data into train and test sets make sure that:
    - Data set is at least large enough to split data randomly. If not, use more advanced sampling methods.
- Discover and viz. data to gain insights
  - Vizualize data w.r.t labels to gain insight (e.g for spatial data: plot using shapes, size and colors to vizualize labels and potential link with other attributes)
  - For numerical attributes: try to find simple correlations between attributes (using DataFrame's corr() or scatter_matrix)
  - Maybe try attribute combination (e.g proportions)
- Data preparation & cleaning
  - Handle missing values (dropping attribute, dropping tuples, filling missing values following a strategy)
  - Handling text and categorical data.
    - Make that distance notion is applyable for categories values. Use One Hot Encoding. If categories number is large One hot encoding is not suitable since it adds significant ammount of features (may slow down training and degrade performance).
  - Custom transformers: whith Scikit-learn, it is possible to write custom and specific data preparation and cleaning as "transformers" that implement the (fit, transform and fit_transform). This permit to automate the preparation process for new data and still working seamlessly with Scikit-learn. (cf. jupyter notebook)
  - Feature scaling: depends on data. Common ways to perform scaling: min-max scaling, standardization
  - Transformation pipelines: encapsulate a sequence of transformers and a final estimator. When called (fit() function), it calls fit_transform() sequentially on all transformers passing the output of each call as the input of the next call until reaching the final estimator, for which it call fit()
- Model selection and training
  - Train different models (i.e different algorithms) on training data and compare performances using a distance
  - Use cross-validation when possible.
- Fine tuning
  - Tweak hyper-parameters to improve performances
  - Can be time consuming. Use automattion techniques Grid Search / Randomized search
    - With grid search: if the best parameter is the highest/lowest one, one should run a grid search with higher/lower values.
    - Randomized search for hyperparameters is more efficient.
    - The two can also be time consuming.
  - Use ensemble methods
- Evaluation
  - On the test set (that wasn't touched)
  - Do not forget to transform data using same process as for training.
- Presentation
  - Documenting every thing (models, data transformations, cleaning ...) is useful for presentation.
  - Aim: describe data analysis process, why models were chosen, how hyperparameters were set, remarks on data, expected performances ...
- Monitoring and maintaining
  - Monitoring performances of model in production can be tough: what metrics should one use ? what performance threshold should be used ? Dependes on the context.
  - Models need to be re-trained
  - (cf. concept drift: litterature is abondant)

> #### **Scikit-learn design**
>
> The main design principles of Scikit-learn:
>
> - **Consistency**: all object share a consistent and simple interface:
>   - **Estimators**: an object that can estimate some parameters based on a dataset. The estimation is done with the _fit()_ method (takes dataset, labels for supervised learning and other potential hyperparameters)
>   - **Transformers**: estimators that can also transform datasets. The transformation is done using _transform()_ with the dataset as a parameter. _fit_transform()_ equivalent to _fit()_ then _transform()_ (sometimes optimized and runs faster)
>   - **Predictors**: estimators capable of making predictions. Predictions are obtain with _predict()_ taking a dataset of new instances and return dataset of corresponding predictions. _score()_ measures the quality of the predictions given a test set
> - **Inspection**: All estimators hyperparameters are accessible via public instance variables (e.g _imputer.strategy_ gives "median"), and all learned parameters via instance variables with an underscore (e.g \_impter.statistics\_\_ )
> - **Non-proliferation of classes**: Datasets are represented as NumPy arrays or SciPy sparse matrices. Hyperparameters are strings or numbers.
> - **Composition** Existing building blocks are reused as much as possible
> - **Sensible defaults**: Scikit-learn provides reasonable default values for most parameters (quick prototyping)
>
> _Lars Buitinck et al. API design for machine learning software: experiences from the scikit-learn project. arXiv preprint arXiv:1309.0238 (2013)_ https://arxiv.org/pdf/1309.0238.pdf

## Chapitre 3: Classification

### Validation techniques

- Accuracy not always the best way to evaluate a classifier.
- It's better to use a Confusion matrix
- Precision, recall and F1-score. (When training a model, keep in mind objectives when setting targets for precision and recall: depends on the problem.)
- ROC curve
  (see performance measures cheat sheet)

### Binary classification

Binary classification: two classes, each sample has one class

### Multiclass classification

Multiclass classification: multiple classes, each sample has one class
Two strategies:

- One versus Rest (OvR): Train a classifier to recognize a class over all others for each class. For instance, if 10 classes, 10 classifiers are needed. For prediction, running a sample over the n classifiers and selecting the one providing the higherst score. This strategy is often preferred.
- One versus One (OvO): Train a classifier to recognize a class over another class: $\frac{N*(N-1)}{2}$ classifers are needed. To make a prediction, run the sample over all classifiers and select the one winning the most duels. This strategy is prefered if the used algorithm scales poorly: better training numerous classifiers on small training sets than training few over a large training set.

### Multilabel classification

Multilabel classification: multiples classes, a sample may have multiple classes

Example: Assume a multilabel classifier trained to recognize oranges, apples and, bananas. Samples used to train this classifier should have three lables (o, a, b) one for each class. predictions should take the form of (1, 0 , 1) meaning : image contains an orange, no apples and a banana.

### Mutlioutput classification

## Chapter 4: Training Models

### Linear Regression

### Gradient Descent

### Polynomial Regression

### Learning Curves

### Regularized Linear Models

Regularization -> reducing degrees of freedom. For linear models -> tweaking the cost function so that weights are constrained during training (adding a term to the cost function so that any change in weights contributes greatly to gradient)

#### Ridge Regression

Also called _Tickonov regularization_ is a regularized version of of Linear Regression. It consists of adding a _regularization_ term equal to $\alpha \sum_{i=1}^n \theta_i^2$ to the cost function.

$\alpha$ allows to control how much you learning is regularied. If $\alpha = 0 $, then Ridge Regression is just Linear Regression. If $\alpha$ is very big, then all weights end up very close to 0 and the result is a flat line goind through the data mean.

**Ridge Regression Cost Function**
$$J(\theta) = MSE(\theta) + \alpha \frac{1}{2} \sum_{i=1}^n \theta_i^2 $$
Note that the biais term is not included in the regularization term.

The regularization term should only be added to cost function during training. Model evaluation should be done using a unregulraized performance measure.

This regularization can also be used with Stochastic Gradient Descent (l2 regularization).

Ridge regression -> Reduces weights that are close to 0

#### Lasso regression

Lasso (Least Absolute Shrinkage and Selection Operator Regression). The same as Ridge Regression except the feature weights are not squared in the regularization sum.

Lasso regression may permit to perform feature selection. see https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b#:~:text=Ridge%20and%20Lasso%20regression%20are,the%20magnitude%20of%20the%20coefficients.

#### Elastic Net

Elastic Net is a midddle ground between Ridge and Lasso.

**Elastic Net cost function**
$$J(\theta) = MSE(\theta) + \alpha (r\sum_{i=1}^n \theta_i + \frac{1-r}{2}\sum_{i=1}^n \theta_i^2) $$

#### When to use Ridge, Lasso, Elastic or plain linear regression ?

Usually it's always better to use a bit of regularization. Ridge Regression is a good default unless suspeting that very few features are useful, then go for Elastic or Lasso because then tend to reduce the useless features' weights down to zero. Elastic Net is generally preferred over Lasso because Lasso can behave erratically when number of features is greater that the number fo traning instances or when several features are strongly correleted.

#### Early Stoping

Stopping the training as soon as validation errors reach a minimum. "Beautiful free lunch".

Validation error starts by decreasing with the training and then increase. The idea here is that the increase indicates that the model starts to overfil the training data, thus, training should be stopped.

The difficulty using this approach is to determine a condition to decide if the increase for validation error is "real" or resulting of some variations (non-smooth). In this case, one solution is to stop training after the validation error is above the minimum continuously for some time.

### Logistic Regression

Is actually a binary classifier.
