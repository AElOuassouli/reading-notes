# Cross-Validation

## What is Cross-Validation (CV) ? 

## When to use (CV) ? 

**Pre-evaluation of models**: Before testing a model on the test dataset, cross-validation can be useful to obtain insight on performances of a trained model.
**Model selection**: Compare performances of different models 
**Fine tuning**: 

## Types
### *K-fold* cross-validation
**Principle**: 
1. Randomly splits a dataset into $k$ distinct subsets, named *folds*.
2. Train and evaluate $k$ times, picking a different fold for evaluation and training on the other $k-1$ folds.
3. Provides $k$ evaluation scores. 

See mean and standard deviation of provided scores to evaluate performances.

**Scikit-learn function**: [sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)

### 