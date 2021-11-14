
#Reading notes of: Hands-on machine learning with Scikit-learn, Keras & Tensorflow (Aurélien Géron)

##Chapter 1: The machine learning landscape. 

#####General definitions of ML: 
>The field of study that gives computers the ability to learn withour being explicilty programmed.  
Arthur Samuel (1959)

>A computer program is said to learn from experience $E$ with respect to some task $T$ and some performance measure $P$, if its performance on $T$, as measured by $P$, improves with $E$.
Tom Mitchell (1997) 


#####Why using machine learning? 
In constrast with hard-coding a set of rules defining a given behaviour, machine learning is devised to extract models from data permitting to mimic the behaviour. It can be used for: 
* Simple problems requiring a lot of fine tuning and maintenance (updating frequently rules). 
* Complex problems (or complex behaviours) that cannot be solved traditionally by an enumaration of rules. (e.g image)
* Dynamic structure: behaviours in the environment are evolving. The system needs to update the set of rules describing the behaviour. 
* Knowledge extraction: gaining insight about an environement/subject by studying structures provided by ML algorithms. (Data Mining) 


#####Types of machine learning

Classification of ML algorithms w.r.t 3 criteria: Supervision, Online/batch learning and, Instance-based/Model-based learning

######Supervision: What amount and type of supervision do an algorithm require ? 

1. **Supervised algorithms**: The training data contain the desired outcom, called *labels*. Data can be represented as a function:  $f(X) = y$ where $X$ is a vector of *attributes* and $y$ the label. The problem is to learn $f$. Some supervised algorothms: k-nearest neighbors, regressions, SVM, decision trees, random forests, neural networks (not all of them)
2. **Unsupervised learning**: Training data do not contain labels. Data sets can be represented as a set of vectors $X$. The problem here is to find structure in the data ; e.g. are they in different groups ? is there correlations between attrbutes ? Some unsupervised tasks: clusturing (K-Means, DBSCAN, Hierachical cluster analysis), Anomaly and novelty detection, Visualisation and dimensionality reduction (Principal Component Analysis, Kernel PCA, locally linear embedding), association rule leaning (Apriori, Eclat)
3. **Semi-supervised learning**: dealling with partially labeled data. Algorithms in that category generally mixe supervised and unsupervised learning.
4. **Reinforcement learning**: The learning system do not really have training data but observes an environment, make actions (or predictions), gets rewarded or not and updates itself based on the reward or penalty. It learns by itself what is the best strategy by experience over time. The problem here is to actually be able to *converge* and define the best strategy to do so quickly (updating function besed on *rewards/penalties*) (Note to self: read more about reinforcement learning and game theory)

######Batch and online learning: Can an algorithm upadate its model easily? 
Very succintly, batch learning use availbale data in batches: algorithms learn a model once and do not update it. Updating models requires to re-run algorithms. Online learning algorithms update models with novel and incoming data incrementaly and sequentially using data nuggets or *mini-batches*. 

######Instance-based vs. Model-based learning: how do algorithm *generalize* ?

* Is generalization done by similarity ? Use of a similarity measure to compare new data to known data. If a given data point is similar w.r.t to a similarity measure (a distance in a given space), to known data points.  
* Is generalization done by prediction given a model/representation ? New data points are processed and predictions are done using models (equations, graphs ...). (e.g regression)

#####Main challenges of machine learning
* Insuficient quantitiy of data
* Non representative training data. when data do not represent new cases to be generalized to (sampling biais)
* Poor-quality data: errors, outliers, amount of noise, missing attrbutes
* Irrelevant features (feature engineering)
* Overfitting training data: performing very good with training data but not being able to generalize (e.g patterns in noise). (regularization) 
* Underfitting: model is too simple to effectively learn from data. *Pattern language's expressivity does not permit to capture a pattern complexity*. 

#####Testing and validating
Spliting data sets into training set and test set. Common rule: use 80% of data for training (but depends on the training set size: if 10 million instances, 1% for testing is probably enough). 

**Training error (TE)**: errors that a model makes with training set. 
**Generalization error (GE)**: errors that a model makes with new instances. 

If GE low and TE high: overfitting. 

**Holdout validation**: Hold out a part of the training set (validation set, development set, dev set) to evaluate several models (i.e linear, logistic ...) with different hyperparameters. Chose the best model and train it on the entire training set (including the validation set). Then, evaluate *genealization* on the test set.  

**Data mismatch**: The validation and test sets must be as representative as possible data to expect in production. The objective is perform well for the expected data. Thus, you can train models on various data sets by always validate and test generalization on representative data sets. 

**No free lunch theorem** (Copied from http://www.no-free-lunch.org/. accessed 24/10/2021)

Hume (1739–1740) pointed out that ‘even after the observation of the frequent or constant conjunction of objects, we have no reason to draw any inference concerning any object beyond those of which we have had experience’. More recently, and with increasing rigour, Mitchell (1980), Schaffer (1994) and Wolpert (1996) showed that bias-free learning is futile.

Wolpert (1996) shows that in a noise-free scenario where the loss function is the misclassification rate, if one is interested in off-training-set error, then there are no a priori distinctions between learning algorithms.

More formally, where
* $d$ = training set;
* $m$ = number of elements in training set;
* $f$ = ‘target’ input-output relationships;
* $h$ = hypothesis (the algorithm's guess for f made in response to d); and
* $C$ = off-training-set ‘loss’ associated with f and h (‘generalization error’)

all algorithms are equivalent, on average, by any of the following measures of risk: $E(C|d), E(C|m), E(C|f,d),$ or $E(C|f,m)$.

How well you do is determined by how ‘aligned’ your learning algorithm $P(h|d)$ is with the actual posterior, $P(f|d)$.

Wolpert's result, in essence, formalizes Hume, extends him and calls the whole of science into question.

##Chapter 2: End-to-end Machine Learning Project

Basic steps when dealing with a ML problem (notes correspond mostly to a regression problem): 
* Look at the big picture
  * Decide wich type of ML problem (regression, classification) (supervised, unsupervised) (what type of data and what type of outcome)
  * Select a performance measure
  * Check assumptions
* Get the data
  * Always get insights (e.g historgrams) about data before doing anything (no free lunch theorem). Permits also to spots some elements to consider when preparing data for ML (spotting outliers, attributes unit ...) and chose an ML model. Permits also to spot some potential biaises.    
  * When spliting data into train and test sets make sure that:
    * Data set is at least large enough to split data randomly. If not, use more advanced sampling methods. 
* Discover and viz. data to gain insights
  * Vizualize data w.r.t labels to gain insight (e.g for spatial data: plot using shapes, size and colors to vizualize labels and potential link with other attributes)
  * For numerical attributes: try to find simple correlations between attributes (using DataFrame's corr() or scatter_matrix)
  * Maybe try attribute combination (e.g proportions) 
* Data preparation & cleaning
  * Handle missing values (dropping attribute, dropping tuples, filling missing values following a strategy)
  * Handling text and categorical data. 
    * Make that distance notion is applyable for categories values. Use One Hot Encoding. If categories number is large One hot encoding is not suitable since it adds significant ammount of features (may slow down training and degrade performance).
  * Custom transformers: whith Scikit-learn, it is possible to write custom and specific data preparation and cleaning as "transformers" that implement the (fit, transform and fit_transform). This permit to automate the preparation process for new data and still working seamlessly with Scikit-learn. (cf. jupyter notebook)
  * Feature scaling: depends on data. Common ways to perform scaling: min-max scaling, standardization
  * Transformation pipelines: encapsulate a sequence of transformers and a final estimator. When called (fit() function), it calls fit_transform() sequentially on all transformers passing the output of each call as the input of the next call until reaching the final estimator, for which it call fit()      
* Model selection and training
* Fine tuning 
* Presentation 
* Monitoring and maintaining



> #### **Scikit-learn design**
> The main design principles of Scikit-learn: 
> * **Consistency**: all object share a consistent and simple interface:
>   * **Estimators**: an object that can estimate some parameters based on a dataset. The estimation is done with the *fit()* method (takes dataset, labels for supervised learning and other potential hyperparameters)
>   * **Transformers**: estimators that can also transform datasets. The transformation is done using *transform()* with the dataset as a parameter. *fit_transform()* equivalent to *fit()* then *transform()* (sometimes optimized and runs faster)   
>   * **Predictors**: estimators capable of making predictions. Predictions are obtain with *predict()* taking a dataset of new instances and return dataset of corresponding predictions. *score()* measures the quality of the predictions given a test set 
> * **Inspection**: All estimators hyperparameters are accessible via public instance variables (e.g *imputer.strategy* gives "median"), and all learned parameters via instance variables with an underscore (e.g *impter.statistics_* )  
> * **Non-proliferation of classes**: Datasets are represented as NumPy arrays or SciPy sparse matrices. Hyperparameters are strings or numbers. 
> * **Composition** Existing building blocks are reused as much as possible
> * **Sensible defaults**: Scikit-learn provides reasonable default values for most parameters (quick prototyping)  
> 
> *Lars Buitinck et al. API design for machine learning software: experiences from the scikit-learn project. arXiv preprint arXiv:1309.0238 (2013)* https://arxiv.org/pdf/1309.0238.pdf

