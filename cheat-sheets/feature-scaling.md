# Feature scaling techniques

Cheat sheat on feature scaling. 

#### Useful notations definitions

$X$: a dataset (a matrix of features x instances)
$N$: number of instances in $X$ 
$A$: a feature of $X$
$a:$ a value of $A$ to be rescaled
$a^*$: rescaled value corresponding to $a$


#### Min-max (a.k.a Normalization)
Shifts values and rescaled to range in [0,1]. 
$$a^* = \dfrac{a - min}{max - min}$$ where $min$ ($max$) is the minimum (maximum) value of feature $A$ in $X$.
**Key caracteristics**: Can be affected significantly by outliers but provides values in a controlled range.  
**Scikit-learn transformer**: [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) 

#### Standardization 
Transformer dataset such that is has a zero mean and the standard deviation as a unit.
$$ a^* = \dfrac{a - \sigma}{\sigma} $$  where $\sigma$ is the standard deviation of values of $A$. 
**Key caracteristics**: Less affected by outliers in comparison with Normalization. Not bounded. 
**Scikit-learn transformer**: [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)   

