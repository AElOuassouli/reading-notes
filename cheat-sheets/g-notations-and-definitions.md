# Notations & Definitions

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