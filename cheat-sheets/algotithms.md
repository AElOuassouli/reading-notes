#Algorithms

####Definitions & Notations

$X$: a matrix containing all feature values (excluding labels is available) of all instances. Row = instance. It is noted as follows: 
$$ X = \begin{pmatrix}
(x^{(1)})^T \\
(x^{(2)})^T \\
... \\
(x^{(n)})^T
\end{pmatrix} = 
\begin{pmatrix}
a^1 & b^1 & c^1 & ... \\
a^2 & b^2 & c^2 & ... \\
... & ... & ... & ... \\
a^n & b^n & c^n & ...  
\end{pmatrix}
$$ 
where $x^{(i)} = \begin{pmatrix} a^i \\ b^i \\ c^i \\ ...   \end{pmatrix}$ is a vector of values $a^i$, $b^i$, $c^i$ ... for features $a$, $b$, $c$ ... for the $ith$ instance and, $(x^{(i)})^T$ the transpose of $x^{(i)}$.

---

$y = \begin{pmatrix} y^1 \\ y^2 \\ ... \\ y^n   \end{pmatrix}$: the vector of labels corresponding to $X$ where $y^i$ the label of the $ith$ instance $x^{(i)}$

---
$h$: the system's prediction function, or *hypothesis*, or *theory*. It outputs a predicted value $\hat{y}^{(i)} = h(x^{(i)})$


##Regression

###Performance measures

**Root mean square error**: euclidian distance.
$$RMSE(X, h) = \sqrt{\dfrac{1}{m} \sum_{i=1}^{m}   \big(h(x^{(i)}) - y^{(i)} \big)^2 }$$ 

**Mean absolute error**: Manhattan distance.
$$MAE(X, h) =  \dfrac{1}{m} \sum_{i=1}^{m} \mid h(x^{(i)})- y^{(i)} \mid $$

Any norm $k$, noted $\mid\mid . \mid\mid_k$, can be used. The higher the norm, the more it focuses on large values and neglects small ones. Higher order norms are more sensisble to outliers.   **Warnings**: May be source of algorithmic biais. choose carefully.   