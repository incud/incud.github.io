# Implementation of quantum kernels with Qiskit and scikit-learn

<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

Supervised learning is a vital aspect of machine learning, enabling computers to learn from labeled data and make predictions. Kernel methods offer a powerful solution for addressing complex learning problems by transforming data into higher-dimensional spaces. These methods leverage kernel functions, which can be classical or quantum, to capture intricate relationships between variables. Classical kernels, like Gaussian or polynomial, are widely used, while quantum kernels tap into quantum computing's potential.

## Introduction to scikit-learn

Scikit-learn is a widely used Python library for machine learning, offering a rich set of tools and algorithms. It provides a user-friendly interface for tasks like classification, regression, clustering, and dimensionality reduction. With its extensive documentation and ease of use, scikit-learn is a popular choice for ML practitioners and researchers.

Some useful links:
* Supervised learning models: [here](https://scikit-learn.org/stable/supervised_learning.html)
* Distance and kernel functions: [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise)
* Datasets: [here](https://scikit-learn.org/stable/datasets.html)



```python
import sklearn
sklearn.__version__
```




    '1.1.1'



### A toy example

we load the Diabetes dataset, a well-known benchmark dataset in machine learning. The scikit-learn's Diabetes dataset consists of various physiological and health-related features of individuals, aiming to predict the numerical measure of disease progression, providing valuable insights for diabetes management and research.

Next, we instantiate a Support Vector Machines (SVM), which is a machine learning algorithm that separates data points into different classes by finding the optimal hyperplane in a high-dimensional feature space that maximally separates the classes, with the help of support vectors, which are the closest data points to the hyperplane.

Finally, we calculate the accuracy of our model by comparing the predicted outputs with the true labels from the dataset, providing us with a quantitative measure of its performance in accurately predicting the species of Iris flowers based on the given features.


```python
from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Iris dataset
MAX_ELEMENTS = 30
MAX_FEATURES = 4
iris = load_diabetes()
X = iris.data[:MAX_ELEMENTS, :MAX_FEATURES]
y = iris.target[:MAX_ELEMENTS]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33334, random_state=5454)

# Instantiate a machine learning model
model = LinearRegression()
# model = LinearRegression(fit_intercept=False)
# model = KernelRidge()
# model = SVR()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = model.predict(X_test)

# Calculate the accuracy
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
rmse, r2
```




    (72.33599613368506, -0.18011566958984604)



The purpose of this example is **not** improve the performance of the model with quantum kernels, but to introduce the software machinary to implement them. 

Looking for use cases? See:
* Cerezo, M., Verdon, G., Huang, HY. et al. Challenges and opportunities in quantum machine learning. Nat Comput Sci 2, 567–576 (2022). https://doi.org/10.1038/s43588-022-00311-3
* Woźniak A., Belis, V.,  Puljak, E. et al. "Quantum anomaly detection in the latent space of proton collision events at the LHC." arXiv preprint arXiv:2301.10780 (2023).

### How to construct a kernel Gram matrix

To train a kernel machine on a training set ${ (x^{(i)}, y^{(i)}) }_{i=1}^n$ using a kernel $\kappa$, it is necessary to compute the Gram matrix denoted as $K$, which is calculated as follows:
$$K = [\kappa(x^{(i)}, x^{(j)})]_{i,j=1}^n.$$


```python
def kappa(xi, xj):
    return np.exp(-0.1 * np.linalg.norm(xi - xj))

def build_gram_matrix(XA, XB, k):
    return np.array([[k(xi, xj) for xj in XB] for xi in XA])
    # import scipy
    # return scipy.spatial.distance.cdist(XA, XB, metric=k)
```


```python
K_train = build_gram_matrix(X_train, X_train, kappa)
K_test = build_gram_matrix(X_test, X_train, kappa)

model = SVR(kernel='precomputed')
model.fit(K_train, y_train)
y_pred = model.predict(K_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse
```




    72.69927956684273



## The first quantum kernel with Qiskit

To define a quantum kernel using Qiskit, you can leverage the power of the machine learning plugin. To install that you can use the command:


```python
# !pip install qiskit_machine_learning --upgrade
```

Please be aware that this command differs from what is outlined in the documentation. However, it ensures the installation of the most recent package version, which you can confirm by using:


```python
import qiskit_machine_learning
qiskit_machine_learning.__version__
```




    '0.6.1'



To proceed, it is necessary to establish the feature map, which is essentially a parametric quantum circuit. The number of parameters within the feature map should correspond to the number of features present in the dataset. 

Typically, the number of qubits within the circuit corresponds to the number of features as well, although this requirement is not strictly mandatory.


```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

params = ParameterVector('x', MAX_FEATURES)
feature_map = QuantumCircuit(4)
feature_map.rx(params[0], 0)
feature_map.rx(params[1], 1)
feature_map.rx(params[2], 2)
feature_map.rx(params[3], 3)
feature_map.cx(0, 1)
feature_map.cx(1, 2)
feature_map.cx(2, 3)
feature_map.rx(params[0], 0)
feature_map.rx(params[1], 1)
feature_map.rx(params[2], 2)
feature_map.rx(params[3], 3)
feature_map.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">     ┌──────────┐     ┌──────────┐                        
q_0: ┤ Rx(x[0]) ├──■──┤ Rx(x[0]) ├────────────────────────
     ├──────────┤┌─┴─┐└──────────┘┌──────────┐            
q_1: ┤ Rx(x[1]) ├┤ X ├─────■──────┤ Rx(x[1]) ├────────────
     ├──────────┤└───┘   ┌─┴─┐    └──────────┘┌──────────┐
q_2: ┤ Rx(x[2]) ├────────┤ X ├─────────■──────┤ Rx(x[2]) ├
     ├──────────┤        └───┘       ┌─┴─┐    ├──────────┤
q_3: ┤ Rx(x[3]) ├────────────────────┤ X ├────┤ Rx(x[3]) ├
     └──────────┘                    └───┘    └──────────┘</pre>



We define explicitly the quantum kernel as the inner product of the quantum states $|x_i\rangle, |x_j\rangle$ encoded using the feature map. We can implement the calculation of the inner product in several ways, the most straightforward is throught the fidelity test telling that $\langle x_i|x_j\rangle$ equal to probability of outcome $0^n$ after measuring in the computational basis the system evolved using .

The quantum kernel is explicitly defined as the inner product of the quantum states $|x_i\rangle$ and $|x_j\rangle$, which are encoded using the feature map. There are various methods available to calculate this inner product, with the simplest approach involving the fidelity test. According to the fidelity test, $\langle x_i|x_j\rangle$ is equal to the probability of obtaining the outcome $0^n$ after measuring the system, evolved using the operator $U^\dagger(x_j)U(x_i)$, in the computational basis.


```python
def quantum_kappa(xi, xj, fm):
    n = fm.num_qubits
    qc = QuantumCircuit(n, n)
    qc.append(fm.bind_parameters({params: xi}), range(n))
    qc.append(fm.inverse().bind_parameters({params: xj}), range(n))
    qc.measure(range(n), range(n))
    backend = Aer.get_backend('qasm_simulator') 
    SHOTS = 100_000
    counts = execute(qc, backend, shots=SHOTS).result().get_counts()
    inner_product = counts.get('0' * n, 0) / SHOTS
    return inner_product
    
kij = quantum_kappa(X[0], X[1], feature_map)
kij
```




    0.07013



To streamline the process and minimize the chances of errors in the code, we have the option to utilize the `QuantumKernel` class, which eliminates the need for explicit instantiation of the quantum circuit. This automation is highly beneficial. The `QuantumKernel` class provides a convenient solution by encapsulating the necessary functionalities. By calling its evaluate method, we can directly access the corresponding function κ, allowing for a more efficient and reliable implementation.


```python
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.utils import QuantumInstance

qi = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=100_000)
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=qi)
quantum_kernel.evaluate(X[0], X[1])
```

    /tmp/ipykernel_166/1527164809.py:4: DeprecationWarning: The class ``qiskit.utils.quantum_instance.QuantumInstance`` is deprecated as of qiskit-terra 0.24.0. It will be removed no earlier than 3 months after the release date. For code migration guidelines, visit https://qisk.it/qi_migration.
      qi = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=100_000)





    array([[0.06963]])



In order to adapt to the evolving capabilities of Qiskit, the QuantumKernel class has been deprecated and replaced with a new API that leverages the Qiskit Runtime framework. 


```python
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute

sampler = Sampler()
fidelity = ComputeUncompute(sampler)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
quantum_kernel.evaluate(X[0], X[1])
```




    array([[0.0692783]])



## Some properties of quantum kernels

When studying quantum kernels, it is crucial to analyze their properties to gain insights into their behavior and applicability. Here, we highlight three important aspects of quantum kernels: eigenvalue distribution, concentration of values, and kernel alignment.

### Eigenvalue distribution

The eigendecomposition of the kernel matrix $K$ can provide insights into our procedure. Such eigendecomposition approximates the eigendecomposition of $\kappa$ for a large number of data points approaching infinity.

According to the framework of (Canatar et al. 2021), the components (eigenvectors for $K$, eigenfunctions for $\kappa$) corresponding to large eigenvalues can be learned easily with fewer data points, while components with smaller eigenvalues are more difficult to learn. In the extreme case, a component with a zero eigenvalue cannot be learned (out-of-RKHS component). A desirable situation occurs when the eigenvalue distribution is non-flat and decays, either polynomially or exponentially fast. A decaying eigenvalue distribution suggests that the kernel may be effective depending on the dataset, while flat distributions result in useless kernels.

See:
* Canatar, Abdulkadir, Blake Bordelon, and Cengiz Pehlevan. "Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks." Nature communications 12.1 (2021): 2914.
* Kübler, Jonas, Simon Buchholz, and Bernhard Schölkopf. "The inductive bias of quantum kernels." Advances in Neural Information Processing Systems 34 (2021): 12661-12673.


```python
from numpy.linalg import eigh

K_train = quantum_kernel.evaluate(X_train, X_train)
eigvals, eigvecs = eigh(K_train)
```


```python
np.sort(eigvals)[::-1]
```




    array([4.20467968, 2.32811921, 1.99754652, 1.72120043, 1.67396274,
           1.18445889, 1.10629552, 0.9352618 , 0.80588921, 0.65373221,
           0.54892027, 0.5081392 , 0.43884821, 0.38535748, 0.19377373,
           0.1783879 , 0.08000036, 0.04425475, 0.01117187])



A flat distribution of eigenvalues will likely occur when dealing with circuits containing more than 10 qubits and "expressible" feature maps. In order to mitigate this, one can reduce the expressiveness of the feature map, leading to a non-flat eigenvalue distribution. One possible solution is to introduce a bandwidth parameter $\beta \in [0,1]$, which acts as a rescaling constant to limit the reach of the feature map within the Hilbert space. This approach can help restore a non-flat distribution. 

However, it is important to note that achieving a non-flat distribution does not necessarily guarantee improved performance. The efficacy of the kernel is highly dependent on its compatibility with the task at hand, as indicated by the concept of kernel alignment.

#### Feature map with bandwidth

The bandwidth can be easily implemented as follows. The best bandwidth can be found empirically though hyperparameter optimization. 


```python
bandwidth = 0.1
bw_feature_map = QuantumCircuit(4)
bw_feature_map.rx(bandwidth * params[0], 0)
bw_feature_map.rx(bandwidth * params[1], 1)
bw_feature_map.rx(bandwidth * params[2], 2)
bw_feature_map.rx(bandwidth * params[3], 3)
```

### Concentration of values

Problematic kernels exhibit an unfavorable property known as the concentration of values. This aspect delves into more advanced concepts, and for a thorough and rigorous treatment, I recommend referring to the following resource:
* Thanasilp, Supanut, et al. "Exponential concentration and untrainability in quantum kernel methods." arXiv preprint arXiv:2208.11060 (2022).

This phenomenon occurs when the quantum kernel $\kappa(x_i, x_j)$ exhibits the following behavior:
$$ \kappa(x_i, x_j) \approx \begin{cases}
1, & \text{if } i = j \\
c, & \text{if } i \neq j
\end{cases}$$
where $c$ is a constant. This property is particularly relevant when dealing with expressive circuits that operate in the high-dimensional Hilbert space of the quantum system. When encoding two samples in this space, it becomes relatively easy to generate vectors with inner products close to zero (generally close to $c$ when allowing for some projection).


```python
x1 = np.random.normal(size=(3,))
x2 = np.random.normal(size=(3,))
x1 /= np.linalg.norm(x1)
x2 /= np.linalg.norm(x2)
print(f"Inner product in 3 dim: {x1.dot(x2):0.9f}")
```

    Inner product in 3 dim: 0.611812623



```python
x1 = np.random.normal(size=(3000,))
x2 = np.random.normal(size=(3000,))
x1 /= np.linalg.norm(x1)
x2 /= np.linalg.norm(x2)
print(f"Inner product in 3000 dim: {x1.dot(x2):0.9f}")
```

    Inner product in 3000 dim: 0.007858363


To assess whether the model exhibits concentration of values, one can examine the standard deviation or the variance of the coefficients. A low variance indicates that more precise measurements are required to differentiate between the coefficients, which in turn necessitates a larger number of shots in order to obtain accurate results.


```python
upper_triangular_without_diagonal_indices = np.triu_indices_from(K_train, k=1)
coefficients = K_train[upper_triangular_without_diagonal_indices]
np.std(coefficients)
```




    0.18072980917433412



Again, the introduction a bandwidth parameter can help mitigate this phenomenon, but its effectiveness in improving performance is not guaranteed.

### Kernel alignment

We have said that a given kernel can perform goodly or badly depending on the task at hand (see No Free Lunch theorem). The quantification of how good or bad a kernel behaves for a certain task is its _kernel compatibility_. One way to quantify that is due to the _kernel-target alignment_, which is the normalized inner product between $K$ and $Y = \mathbf{y}^\top \mathbf{y}$:


```python
def target_alignment(K, Y): 
    norm = np.sqrt(np.sum(K * K) * np.sum(Y * Y))
    inner_product = np.sum(K * Y) / norm
    return inner_product

Y_train = np.outer(y_train, y_train)
print(f"The dimensionality of Y is {Y_train.shape}")
```

    The dimensionality of Y is (19, 19)



```python
target_alignment(K_train, Y_train)
```




    0.539815140305761



We aim for the value of the coefficients to be as close to 1 as possible. 

Note that there are additional criteria to evaluate the kernel compatibility, including the _centered kernel alignment_ and the _task model alignment_. These criteria provide further insights into the alignment between the kernel and the task or model at hand.

## Trainable kernels

Two techniques are commonly used to define a family of kernels with adjustable parameters. The first technique involves training the excess parameters using gradient descent to minimize a loss function such as MSE. The second technique involves selecting gates based on specific criteria to design the kernel.

### Gradient descent training of some angle

Define a quantum circuit that depends on both a vector of feature parameters and a vector of trainable parameters. To instantiate the optimization problem and train the quantum circuit, you can use the `TrainableFidelityQuantumKernel` object. 


```python
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel

feature_params = ParameterVector("x", 4)
training_params = ParameterVector("θ", 1)

feature_map = QuantumCircuit(4)
feature_map.rx(training_params[0] * params[0], 0)
feature_map.rx(training_params[0] * params[1], 1)
feature_map.rx(training_params[0] * params[2], 2)
feature_map.rx(training_params[0] * params[3], 3)

quantum_kernel = TrainableFidelityQuantumKernel(feature_map=feature_map, training_parameters=training_params)
```

Such object can then be trained accordingly to your dataset.


```python
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer

spsa_opt = SPSA(maxiter=10, learning_rate=0.05, perturbation=0.05)

qkt = QuantumKernelTrainer(
    quantum_kernel=quantum_kernel, 
    loss="svc_loss", 
    optimizer=spsa_opt, 
    initial_point=[0.1]
)

qka_results = qkt.fit(X_train, y_train)

optimized_kernel = qka_results.quantum_kernel
```


```python
print(qka_results)
```

    {   'optimal_circuit': None,
        'optimal_parameters': {ParameterVectorElement(θ[0]): -2.895568054260639},
        'optimal_point': array([-2.89556805]),
        'optimal_value': -6.534324822420988,
        'optimizer_evals': 20,
        'optimizer_result': None,
        'optimizer_time': None,
        'quantum_kernel': <qiskit_machine_learning.kernels.trainable_fidelity_quantum_kernel.TrainableFidelityQuantumKernel object at 0x7fcd94875520>}



```python

```
