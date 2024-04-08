# PCA notes

Author: Sabina Askerova, Data Science Graduate Student at the University of Strasbourg

Date: 21/03/2024

This page contains some of my notes on Principal Component Analysis (PCA) I took while applying it to a signal time series dataset as part of my research internship. It also contains my explanations regarding mathematical basics necessary for the PCA. 

I cannot express enough how important it is to understand what is happening behind the machine learning library that you use. Yes, it does the computation for you, but it’s up to you to choose how you will apply it, to what data, and to achieve which results. Otherwise, it’s a waste of time.

---

# ✅ Basic formulas

## Covariance

Covariance is the joint probability of two variables.

Assume x and y the features of the original data. The symbol

$$
\bar{x} \space (resp\space \bar{y})
$$

denotes the mean value for the feature x (resp. y)

$cov(x,x) = 1/(N-1)\sum{(x_{ik}-\bar{x_i})(x_{jk}-\bar{x_j})}$

$cov(x,y) = cov(y,x) = 1/(N-1)\sum{(x_{ik}-\bar{x_i})(y_{jk}-\bar{y_j})}$

$cov(y,y) = 1/(N-1)\sum{(y_{ik}-\bar{y_i})(y_{jk}-\bar{y_j})}$

Covariance matrix (2x2)

$$S = [\begin{smallmatrix}
   cov(x,x) & cov(x,y) \\
   cov(y,x) & cov(y,y)
\end{smallmatrix}]$$

Just keep in mind that we need the covariance values for every pair of the features. 

So, if we have 4 features (e.g. 4 signals from 4 different sources), the size of the covariance matrix will be (4x4).

## Eigenvalue

In the next formula the eigenvalue vector is 

$$
\lambda = (\lambda_1,\lambda_2,...,\lambda_n)
$$

where n is the number of features, for example our 4 signals.

The identity matrix is of (nxn) size (e.g. (4x4)

$$
I_{n} = 
\begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}
$$

Alright, all dimensions are known now. To find the eigenvalues, we use the next formula:

$det(S-\lambda I) = 0$

and we solve the system of linear equations.

We then sort eigenvalues in increasing order. The first eigenvalue (the biggest one) will represent the Principal Component that gives the most of the information on the original data (hence explains the variance in the data in the best way). The second one will give some more information on the variance in addition to the first one, and so on. 

The total of n eigenvalues will provide all the information because they will represent the original data (we’re not interested in this in PCA because we want to reduce the dimension. For instance, we want to see clearer what happens in the multi-origin signal constituted from 4 subsignals. We could take only the first two out of four components that would have some valuable information and simplify the analysis). 

We will see how to choose the number of components a little bit later.

## Eigenvector

To find the eigenvector U1 associated with the first eigenvalue, we solve:

$(S-\lambda_1 I) U_1 = 0$

The same goes for all n eigenvalues. 

We get eigenvectors denoted:

$$
U_1, U_2, ... , U_n
$$

Each eigenvector is of size of n.

Note that when we will choose k, the number of components for our next analysis steps, we will use only the k first eigenvectors to transform our original data (we will project it to the space of chosen components/dimensions).

For applying PCA, it’s important to understand what you want to analyze and to keep the dimensions of your data, eigenvalues, eigenvectors in mind in case you get confused. Once you’re okay with your goal vision and dimensions, there is no problem as to apply it to the signals and their power spectrum (which is a whole another topic).

# Case of simple signals

Let’s take a look at four simple signals.

```python
T = 100  # signal duration in seconds
dt = 0.005  # 200 measures for each second
num_times = int(T / dt)
times = np.linspace(0.0, T - dt, num_times)
freq1 = 1/T  # Frequency of the first signal
x1 = np.sin(2*  np.pi * freq1 * times)
x2 =4* x1 + np.random.normal(0,0.5,num_times)
x3 =np.random.normal(0,0.1,num_times)
x4 =np.random.normal(0,0.1,num_times)
```

As you can see the first one is a sine, the second one too and the last two are noise.

This is what the data will look like 

signals.dat:

```
0.000000 0.945905 0.035828 0.036635
0.000314 0.226824 -0.028569 0.066038
0.000628 -0.192670 0.066110 -0.030629
0.000942 1.022826 0.148640 -0.100505
0.001257 0.581576 -0.213087 0.019193
0.001571 0.354838 -0.055841 -0.029577
0.001885 -0.262860 0.073848 0.052751
0.002199 -0.743501 0.082529 -0.085173
0.002513 -0.513044 0.154027 0.093659
0.002827 -0.014640 0.009284 -0.106479
0.003142 -0.098736 -0.139527 -0.095468
0.003456 -0.246036 -0.106794 0.194409
0.003770 0.849640 0.024673 -0.048199
0.004084 -0.154920 -0.072550 -0.085078

etc.
```

The number of samples (time points) N = T/dt = 100/0.005 = 20000

The number of features(signals) n = 4

Mean values of variables: $\bar x_1, \bar x_2,\bar x_3,\bar x_4, ... \bar x_{20000}$

(a variable in this context is a time point in the time series)

Let’s denote the features : $F = {f_1, f_2, f_3, f_4}$

$$ \text{cov}(x,y) = \frac{1}{N-1}\sum_{k=1}^{N} (x_{k}-\bar{x})(y_{k}-\bar{y}) $$

where $x,y \in F$

The size of covariance matrix S  is (4x4).

To calculate $det(S-\lambda I) = 0$ we use Leibniz formula
<img width="364" alt="Screenshot_2024-03-09_at_12 43 17" src="https://github.com/sabinaaskerova/notes/assets/91430159/164731ff-d489-4a8e-a5b8-373f959bc539">

Linear system with 4 unknowns → we will have 4 eigenvalues which we will sort by descending order (the highest corresponds to PC1 etc.)

$$(S-\lambda_i I) U_i = 0, i \in {1,4}$$ → 4 corresponding eigenvectors

$U_i=(u_1, u_2, u_3 ,u_4)$ for i in {1,2,3,4}

We then normalize eigenvectors.

And we derive a new dataset from them!

For example, the number of PC = 2

Original data:

|  | f1 (signal1) | f2(signal2) | f3(signal3) | f4(signal4) |
| --- | --- | --- | --- | --- |
| x1 | v1_1 | v1_2 | v1_3 | v1_4 |
| … |  |  |  |  |
| x20000 | v20000_1 | v20000_2 | v20000_3 | v20000_4 |

New dataset:

| sample | x1 | … | x20000 |
| --- | --- | --- | --- |
| PC1 | p11 |  | p20000_1 |
| PC2 | p12 |  | p20000_2 |

U1 and U2 contain 4 values

Mi contain 4 values for a sample (4 signals per time point)

(1,4)x(4,1)

 $p_{1\_1} = U_1 M_1$

$p_{1\_2} = U_2 M_1$

$p_{20000,1} = U_1 M_{20000}$

$p_{20000\_2} = U_2 M_{20000}$

$$S = \begin{smallmatrix}
   cov(f1,f1) & cov(f1,f2) & cov(f1,f3) & cov(f1,f4)\\
   cov(f2,f1) & cov(f2,f2) & cov(f2,f3) & cov(f2,f4)\\ 
   cov(f3,f1) & cov(f3,f2) & cov(f3,f3) & cov(f3,f4)\\
   cov(f4,f1) & cov(f4,f2) & cov(f4,f3) & cov(f4,f4)
\end{smallmatrix}$$

In a PC1-PC2 space

$$S = \begin{smallmatrix}
   cov(f1,f1) & cov(f1,f2)\\
   cov(f2,f1) & cov(f2,f2)
\end{smallmatrix}$$

The variance explained by a PC1 is determined by cov(f1,f1)

The variance explained by a PC1 is determined by cov(f2,f2)

The percentage of the variance explained by PC1 = $\frac {cov(f1,f1)} {cov(f1,f1)+cov(f2,f2)}$ 

The percentage of the variance explained by PC2 = $\frac {cov(f2,f2)} {cov(f1,f1)+cov(f2,f2)}$ 

The four signals:
<img width="1075" alt="Screenshot_2024-03-09_at_20 42 45" src="https://github.com/sabinaaskerova/notes/assets/91430159/c1d15b51-e951-4da6-9b30-197da38166ba">

### ‼️ Note:

The data should be systematically ***centered*** (you can just calculate the mean for every feature and then extract it from every value in the column/row corresponding to this feature). As for ***scaling*** (for this you need to divide all values of a feature by its standard deviation in order for everything to have a unit standard deviation) it depends on the nature of your data. 

If your features are of different nature, for instance, have different measure units, some are one-hot-encoded categorical variables, etc. you definitely should scale in order to get better dimensionality reduction.

In our example, the 4 features represent the (sub)signals, hence, having the same uni of measurement, so we don’t scale it.

### Covariance matrix, eigenvalues and eigenvectors (n_components = 2)

Covariance matrix S (n_features x n_features) (4x4) 

```python
print(pca.get_covariance())
```

```
[[ 5.00025000e-01  2.00063380e+00  4.71722002e-04 -3.36095135e-04]
 [ 2.00063380e+00  8.25658896e+00  2.41005060e-03 -1.04250957e-03]
 [ 4.71722002e-04  2.41005060e-03  1.00271115e-02  1.86272074e-05]
 [-3.36095135e-04 -1.04250957e-03  1.86272074e-05  1.01863659e-02]]
```

Eigenvalues (1 X n_components) (1x2) 

(in reality there are 4 eigenvalues but we don’t use the last two because we chose n_components to be equal 2).

```python
print(pca.explained_variance_) 
```

```
[8.70001996 0.01442063]
```

Eigenvectors (n_components X n_features) (2x4)

```python
print(pca.components_)
```

```
[[ 2.35739972e-01  9.71816157e-01 -9.37054309e-05  1.20655903e-04]
 [ 9.71751391e-01 -2.35722573e-01  1.00039133e-02 -5.83303417e-03]]
```

### How do we transform the data (idem calculate the principal components)?

It’s just the matrix multiplication from what we computed earlier!

$$
PC_1 = x_i*U_1,  , i \in {1,…,20000}
$$

$$
PC_2 = x_i*U_2, i \in {1,…,20000}
$$

```python
n_comp = pca.n_components_
data_new = data_centered@pca.components_.T
for i in range(n_comp):
    plt.plot(range(num_times), data_new[:,i]) # num_times = 20000
    plt.legend(['PC1', 'PC2'])
plt.show()
```
![Screenshot from 2024-03-26 13-42-13](https://github.com/sabinaaskerova/notes/assets/91430159/69769507-029a-4eb7-b103-ea634f890cb8)



```
data = np.loadtxt('signals.dat')

data_centered = data - np.mean(data, axis=0)

pca = PCA(n_components=4)

pca_data = pca.fit_transform(data_centered)
```

```
n_components = range(1, len(pca.explained_variance_) + 1)
plt.figure(figsize=(6, 6))
plt.plot(n_components, pca.explained_variance_, 'o-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.show()
```
<img width="889" alt="scree1" src="https://github.com/sabinaaskerova/notes/assets/91430159/e09d5933-4118-4f87-9c9a-2b19c9877e57">

Using the *elbow criteria*, we choose n_components=2.

Using elbow criteria means that we try to search for a point in the scree plot after which the gain in explain variance (useful information) is negligible. 

We see that after the point 2 we get no explained variance. It is only logical because the rest of the two signals are just noise! (Remember, we defined them with np.random.normal function from numpy package of Python).

Let me show you an example of a scree plot showing the gain in explained variance with each component based on the real signal dataset (containing 16 different signals).

```python
data = np.loadtxt(pre_path)
data_centered = data - np.mean(data, axis=0)

pca = PCA(n_components=16)
pca_data = pca.fit_transform(data_centered)

n_components = pca.n_components
plt.figure(figsize=(6, 6))
plt.plot(n_components, pca.explained_variance_, 'o-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.show()
```
<img width="988" alt="scree2" src="https://github.com/sabinaaskerova/notes/assets/91430159/0524f860-7cc2-4261-803d-47655e73d2b9">

Using the *elbow criteria*, we see that after n_components=2 we don’t get more of significant gain in explained variance. 

We also can look at the cumulated variance to help us decide on the number of components.

Say, you want to keep more than 90% of the useful information from your signal. 

```python
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print("Cumulative explained variance: ", cumulative_variance * 100)
```

```python
Cumulative explained variance:  [ 53.90619801  72.57969598  82.4628387   87.82255165  91.2689811
  94.48915333  96.21887542  97.68436986  98.58758598  99.18755976
  99.53101708  99.70205454  99.84791029  99.92240058  99.98005746
 100.        ]
```

After observing this output, we see that we retain 91.2% of explained variance with 5 components, so we can choose n_components = 5.

# How does PCA work in a nutshell?

- We calculate the covariance between every pair of features
- We then search for values so that the covariance matrix minus these values would make 0
- We want to preserve the initial variance at the maximum while reducing the number of dimensions
- We choose the number of dimensions using elbow criteria or looking at the cumulated variance
- We do a matrix multiplication between the original data and the eigenvectors to get the new vectors in principal components space
- We then use our components for whatever reason we want: visualization, clustering, signal processing methods…

Hope this helps 🙂

Sabina Askerova (Säbina Äsker)
