# First steps in Quantum Programming using Qiskit

<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

## Introduction

Qiskit is an open-source framework for developing quantum computing applications, algorithms, and software. It is designed to be accessible and easy to use for both beginners and experts in quantum computing. The framework is developed by IBM and offers a comprehensive set of tools for quantum circuit design, simulation, and execution on real quantum devices.

The Qiskit framework is built on four main components:
* *Terra*: This is the foundation layer of the Qiskit framework and provides a set of tools for circuit design, simulation, and optimization. It includes a set of quantum gates and operations, tools for quantum state manipulation, and interfaces for working with quantum devices.
* *Aer*: This is the high-performance simulator layer of the Qiskit framework, which allows users to simulate the behavior of quantum circuits on classical computers. It includes a set of tools for simulating noise and errors in quantum systems, and allows for the efficient simulation of large-scale quantum circuits.
* *Runtime*: This is the layer of the Qiskit framework that provides tools for interfacing with quantum hardware and implementation of error mitigation and error correction. It includes a set of tools for measuring and characterizing noise and errors in real quantum devices, as well as tools for implementing error correction codes and protocols.
* *Nature, Finance, Machine Learning, Optimization* (ex *Aqua*): This is the layer of the Qiskit framework that provides tools for developing quantum algorithms and applications. It includes a set of pre-built algorithms and applications, as well as tools for developing custom algorithms and applications.

All the documentation is available [here](https://qiskit.org/documentation/).

## Installation 

The Qiskit platforms requires Python3 with version higher or equal than 3.6.

The default way to install Qiskit is:


```python
!pip install qiskit --upgrade
```


```python
!pip install qiskit_ibm_runtime qiskit_ibm_provider --upgrade
```

The Qiskit platform undergoes updates on a monthly basis, with a list of changes for each version available [here](https://qiskit.org/documentation/stable/0.42/release_notes.html). However, due to frequent changes to the API, updating Qiskit may impact previously written code. As a suggestion, if you have developed your code using Qiskit version XX.X, it is convenient to continue using that version.

You can check the version of Qiskit using:


```python
import qiskit
qiskit.__version__
```




    '0.24.0'



Be careful! Such string can be misleading, because the printed string refers to the qiskit-terra package. To print the version of each package you can use the instruction:


```python
qiskit.__qiskit_version__
```




    {'qiskit-terra': '0.24.0', 'qiskit-aer': '0.12.0', 'qiskit-ignis': '0.4.0', 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.0', 'qiskit-nature': None, 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.4.0', 'qiskit-machine-learning': '0.4.0'}



You can also install additional packages that are not included in the default installation. These are:
* [Qiskit Finance](https://qiskit.org/ecosystem/finance)
* [Qiskit Nature](https://qiskit.org/ecosystem/nature)
* [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning)
* [Qiskit Optimization](https://qiskit.org/ecosystem/optimization)


```python
!pip install qiskit[finance]
```

## Getting started with quantum circuits

The object describing the quantum computation is the quantum circuit. It is described by its number of qubits, number of classical bits, and the sequence of gates applied. Most of the quantum circuit-related API belongs to Qiskit Terra package. 

### Define a quantum circuit

We can define a quantum circuit using the `QuantumCircuit` class. 


```python
from qiskit import QuantumCircuit
```

The initialization of QuantumCircuit objects requires the number of qubits and the number of classical gates. 


```python
qc = QuantumCircuit(2, 1)
qc.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">     
q_0: 

q_1: 

c: 1/
     </pre>



You can also define manually the registers on which the quantum circuit acts, and give them a name.


```python
from qiskit import QuantumRegister, ClassicalRegister
qr = QuantumRegister(2, name='qrx')
cr = ClassicalRegister(1, name='crx')
qc = QuantumCircuit(qr, cr)
qc.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">       
qrx_0: 

qrx_1: 

crx: 1/
       </pre>



You can add more registers later:


```python
qr2 = QuantumRegister(1, name='qry')
qc.add_register(qr2)
qc.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">       
qrx_0: 

qrx_1: 

  qry: 

crx: 1/
       </pre>



### Add quantum gates

Qiskit provides a wide range of quantum gates that can be applied to qubits. Here's a non-exhaustive list of some of the most commonly used gates in Qiskit:

Single-qubit gates:
* Pauli gates: X, Y, Z
* Hadamard gate: H
* Phase gate: S
* $\pi/8$ gate: T
* Identity gate: I

Multi-qubit gates:
* Controlled-NOT (CNOT) gate
* Controlled-Z (CZ) gate
* Swap gate: SWAP
* Controlled-RX (CRX) gate
* Controlled-RY (CRY) gate
* Controlled-RZ (CRZ) gate
* Controlled-U (CU) gate
* Controlled-phase (CPHASE) gate

Qiskit also provides gates for creating specific quantum states, such as:
* Initialize gate: Initialize
* Reset gate: Reset


```python
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">     ┌───┐     
q_0: ┤ H ├──■──
     └───┘┌─┴─┐
q_1: ─────┤ X ├
          └───┘</pre>




```python
qr = QuantumRegister(2)
qc = QuantumCircuit(qr)
qc.h(qr[0])
qc.cx(qr[0], qr[1])
qc.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">       ┌───┐     
q34_0: ┤ H ├──■──
       └───┘┌─┴─┐
q34_1: ─────┤ X ├
            └───┘</pre>



### Add measurements

In Qiskit, measurements are added to a quantum circuit using the measure method of the QuantumCircuit class. When a qubit is measured, the state of the qubit is projected onto one of two possible outcomes (0 or 1) with a probability determined by the state vector of the qubit at the time of measurement. To add a measurement to a circuit, you need to specify which qubit is to be measured and which classical bit to store the measurement result.

It's important to note that measurements in Qiskit are destructive, meaning that the qubit's state is lost after measurement. This is because the measurement process irreversibly collapses the qubit's state into a classical bit value.


```python
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
qr = QuantumRegister(2, 'qr')
cr = ClassicalRegister(2, 'cr')
qc = QuantumCircuit(qr, cr)
qc.h(qr[0])
qc.cx(qr[0], qr[1])
qc.measure(qr, cr)
qc.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">      ┌───┐     ┌─┐   
qr_0: ┤ H ├──■──┤M├───
      └───┘┌─┴─┐└╥┘┌─┐
qr_1: ─────┤ X ├─╫─┤M├
           └───┘ ║ └╥┘
cr: 2/═══════════╩══╩═
                 0  1 </pre>



### Manipulation of quantum circuits

In Qiskit, a quantum circuit can be manipulated using various methods.
* *append*: This method is used to add a gate or a sub-circuit to another circuit.
* *control*: This method is used to add a control to a gate or a sub-circuit.
* *inverse*: This method is used to add the inverse of a gate to a circuit.


```python
qc_x = QuantumCircuit(1, name='a')
qc_x.x(0)

qc_cx = qc_x.control(1)
qc_cx.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">          
q35: ──■──
     ┌─┴─┐
  q: ┤ a ├
     └───┘</pre>




```python
bell = QuantumCircuit(2, name='b')
bell.h(0)
bell.cx(0, 1)
bell.inverse().draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">          ┌───┐
q_0: ──■──┤ H ├
     ┌─┴─┐└───┘
q_1: ┤ X ├─────
     └───┘     </pre>




```python
qc = QuantumCircuit(2)
qc.append(qc_cx, [0, 1])
qc.append(bell.inverse(), [0, 1])
qc.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">     ┌──────┐┌───────┐
q_0: ┤0     ├┤0      ├
     │  c_a ││  b_dg │
q_1: ┤1     ├┤1      ├
     └──────┘└───────┘</pre>



### State initialization


Initialize a quantum state $|0^n\rangle$ 
with the complex amplitudes of $v \in \mathbb{C}^{2^n}$ normalized.


```python
import numpy as np

v = np.array([1, 2, 3, 4])
v = v / np.linalg.norm(v)
qc = QuantumCircuit(2)
qc.initialize(v, [0, 1])
qc.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">     ┌─────────────────────────────────────────────┐
q_0: ┤0                                            ├
     │  Initialize(0.18257,0.36515,0.54772,0.7303) │
q_1: ┤1                                            ├
     └─────────────────────────────────────────────┘</pre>



The resulting circuit is quite complex and contains a reset gate:


```python
qc.decompose().draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">          ┌────────────────────────────────────────────────────┐
q_0: ─|0>─┤0                                                   ├
          │  State Preparation(0.18257,0.36515,0.54772,0.7303) │
q_1: ─|0>─┤1                                                   ├
          └────────────────────────────────────────────────────┘</pre>



To avoid the Reset you can rely on the `StatePreparation` class.


```python
from qiskit.circuit.library import StatePreparation
state_preparation = StatePreparation(v)
qc = QuantumCircuit(2)
qc.append(state_preparation, [0, 1])
qc.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">     ┌────────────────────────────────────────────────────┐
q_0: ┤0                                                   ├
     │  State Preparation(0.18257,0.36515,0.54772,0.7303) │
q_1: ┤1                                                   ├
     └────────────────────────────────────────────────────┘</pre>



### Parametric quantum circuit

Parametric quantum circuit in Qiskit is a quantum circuit that includes one or more parameters, which can be used to represent some unknown value or a set of values. These parameters can be used to create a family of circuits that differ only in the values of the parameters.

Parametric quantum circuits used in quantum machine learning, optimization, and other quantum algorithms that involve optimization or searching for optimal solutions. By varying the values of the parameters, we can explore the space of possible quantum states and find the ones that optimize a given objective function or achieve some other desired property.


```python
from qiskit.circuit import Parameter, ParameterVector
a = Parameter('a')
b = Parameter('b')
vec = ParameterVector('vec', 3)
```

The parameters can be utilized as the angles of the rotational gates.


```python
qc = QuantumCircuit(1)
qc.h(0)
qc.rz(a, 0)
qc.rx(b, 0)
qc.u(vec[0], vec[1], vec[2], 0)
qc.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">   ┌───┐┌───────┐┌───────┐┌─────────────────────────┐
q: ┤ H ├┤ Rz(a) ├┤ Rx(b) ├┤ U(vec[0],vec[1],vec[2]) ├
   └───┘└───────┘└───────┘└─────────────────────────┘</pre>



By binding the value of the free parameters, we can obtain an executable circuit.


```python
qc1 = qc.bind_parameters({a: 0.1, b: 0.2, vec: [0.3, 0.4, 0.5]})
qc1.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">   ┌───┐┌─────────┐┌─────────┐┌────────────────┐
q: ┤ H ├┤ Rz(0.1) ├┤ Rx(0.2) ├┤ U(0.3,0.4,0.5) ├
   └───┘└─────────┘└─────────┘└────────────────┘</pre>



### Look at the state of a quantum circuit (when simulating)

The Statevector class in Qiskit is a representation of the quantum state of a quantum circuit, in the form of a complex vector. It represents the wavefunction of a quantum circuit, which encodes the probability amplitudes of each possible measurement outcome. The Statevector class is useful for a variety of tasks in quantum computing, such as simulating the behavior of a quantum circuit

At the beginning of the computation the system is initialized at $|0\rangle^{\otimes n}$:


```python
from qiskit.quantum_info import Statevector
circuit = QuantumCircuit(2)
ket_00 = Statevector(circuit)
ket_00
```

    Statevector([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                dims=(2, 2))


We can check if that correspond to the math:


```python
import numpy as np
ket_0 = np.array([1, 0])
ket_00 = np.kron(ket_0, ket_0)
ket_00
```




    array([1, 0, 0, 0])



We can use the statevector at any step of the computation


```python
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
ket_00_plus_11 = Statevector(circuit)
ket_00_plus_11
```

    Statevector([0.70710678+0.j, 0.        +0.j, 0.        +0.j,
                 0.70710678+0.j],
                dims=(2, 2))


We can check if that correspond to the math:


```python
ket_1 = np.array([0, 1])
ket_11 = np.kron(ket_1, ket_1)
ket_00_plus_11 = (1/np.sqrt(2)) * (ket_00 + ket_11)
ket_00_plus_11
```




    array([0.70710678, 0.        , 0.        , 0.70710678])



## Simulation & execution of quantum circuits

The simulation and execution of quantum circuit can be achieved using the `execute` function. Its components are:
* _Quantum Circuit_: The execute method requires a quantum circuit as input. This circuit represents the sequence of quantum gates and measurements that we want to run on the backend.
* _Backend_: The execute method also requires a backend to run the circuit on. The backend can be a local simulator, which runs the circuit on a classical computer, or a remote quantum device. Qiskit provides access to various backends, including simulators and actual quantum devices from IBM Quantum.
* _Options_: There are several options that can be specified when calling the execute method. One of the most important is the number of shots, which determines how many times the circuit is run. This is necessary because quantum measurements are probabilistic, and running the circuit multiple times can help to get a more accurate estimate of the measurement probabilities. Other options include specifying the initial state of the qubits, the measurement basis, the optimization level, and whether to use error mitigation techniques.

Once these components are specified, the execute method sends the circuit to the backend for execution. The backend runs the circuit and returns the results, which include the measurement counts and probabilities. These results can then be analyzed and used to obtain various metrics, such as expectation values of observables, or to perform quantum error correction and fault-tolerance protocols.

The first simulator is the `statevector_simulator` in Qiskit Aer, which shares the same functionalities as `Statevector` class in `qiskit.quantum_info`. 


```python
from qiskit import Aer, execute

circuit = QuantumCircuit(2)
backend = Aer.get_backend('statevector_simulator')
job = execute(circuit, backend)
result = job.result().get_statevector()
result
```

    Statevector([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                dims=(2, 2))


The second simulator, `qasm_simulator`, has a behaviour closer to the real devices and samples the solutions form the underlaying probability distribution. 


```python
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()
backend = Aer.get_backend('qasm_simulator')
job = execute(circuit, backend, shots=1000)
result = job.result().get_counts()
result
```




    {'00': 507, '11': 493}



The third simulator, `unitary_simulator`, returns the unitary associated with the given quantum circuit. 


```python
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
backend = Aer.get_backend('unitary_simulator')
job = execute(circuit, backend)
result = job.result().get_unitary()
result
```




    Operator([[ 0.70710678+0.00000000e+00j,  0.70710678-8.65956056e-17j,
                0.        +0.00000000e+00j,  0.        +0.00000000e+00j],
              [ 0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
                0.70710678+0.00000000e+00j, -0.70710678+8.65956056e-17j],
              [ 0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
                0.70710678+0.00000000e+00j,  0.70710678-8.65956056e-17j],
              [ 0.70710678+0.00000000e+00j, -0.70710678+8.65956056e-17j,
                0.        +0.00000000e+00j,  0.        +0.00000000e+00j]],
             input_dims=(2, 2), output_dims=(2, 2))



## Qiskit Runtime

Qiskit Runtime is a quantum computing programming model and service that enables users to optimize their workloads and execute them efficiently at scale on quantum systems. Two primitives, Estimator and Sampler, are available in the initial release of Qiskit Runtime, providing a simplified interface for defining quantum-classical workloads to customize applications.

The *Estimator* primitive performs essential quantum computing tasks by allowing users to efficiently compute and interpret expectation values of quantum operators, which are necessary for many algorithms. Users specify a list of circuits and observables and provide instructions on how to group the lists selectively, resulting in an efficient evaluation of expectation values and variances for a given input parameter.

The *Sampler* primitive takes a user circuit as input and produces an error-mitigated readout of quasiprobabilities. This allows users to evaluate shot results more accurately using error mitigation and enables them to assess the possibility of multiple relevant data points in the context of destructive interference more efficiently.

![title](https://qiskit.org/documentation/partners/qiskit_ibm_runtime/_images/runtime-architecture.png)

Advantages ([more info here](https://qiskit.org/documentation/partners/qiskit_ibm_runtime/compare.html)):

* Primitive interface as abstraction for circuits and variational workload
* Sessions to improve performance for a sequence of jobs
* Abstracted interface that allows for automated error suppression and mitigation

### Sampler

In Qiskit Terra, the Sampler base class sets a standard for user interaction with all Sampler implementations. This simplifies the process of changing the simulator or device used for calculating expectation values, even if the underlying implementation differs.


```python
from qiskit.primitives import Sampler

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

sampler = Sampler()
job = sampler.run(circuit)
result = job.result()
result
```




    SamplerResult(quasi_dists=[{0: 0.4999999999999999, 3: 0.4999999999999999}], metadata=[{}])



This code can be used to run the circuit on the cloud-based IBM Quantum backends.


```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Sampler
from qiskit_ibm_runtime import Options

service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.backend("ibmq_qasm_simulator")
options = Options()
sampler = Sampler(session=backend, options=options)
job = sampler.run(circuit)
```

    /usr/local/lib/python3.9/dist-packages/pkg_resources/__init__.py:123: PkgResourcesDeprecationWarning: 0.23ubuntu1 is an invalid version and will not be supported in a future release
      warnings.warn(
    /usr/local/lib/python3.9/dist-packages/pkg_resources/__init__.py:123: PkgResourcesDeprecationWarning: 0.1.36ubuntu1 is an invalid version and will not be supported in a future release
      warnings.warn(



```python
print(f">>> Job ID: {job.job_id()}")
print(f">>> Job Status: {job.status()}")
```

    >>> Job ID: chn6lorn6lo4mve4v7ig
    >>> Job Status: JobStatus.QUEUED



```python
result = job.result()
print(f">>> {result}")
print(f"  > Quasi-distribution: {result.quasi_dists[0]}")
print(f"  > Metadata: {result.metadata[0]}")
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Input In [61], in <cell line: 1>()
    ----> 1 result = job.result()
          2 print(f">>> {result}")
          3 print(f"  > Quasi-distribution: {result.quasi_dists[0]}")


    File ~/.local/lib/python3.9/site-packages/qiskit_ibm_runtime/runtime_job.py:221, in RuntimeJob.result(self, timeout, decoder)
        219 _decoder = decoder or self._final_result_decoder
        220 if self._results is None or (_decoder != self._final_result_decoder):
    --> 221     self.wait_for_final_state(timeout=timeout)
        222     if self._status == JobStatus.ERROR:
        223         error_message = self.error_message()


    File ~/.local/lib/python3.9/site-packages/qiskit_ibm_runtime/runtime_job.py:310, in RuntimeJob.wait_for_final_state(self, timeout)
        306     self._ws_client_future = self._executor.submit(
        307         self._start_websocket_client
        308     )
        309 if self._is_streaming():
    --> 310     self._ws_client_future.result(timeout)
        311 # poll for status after stream has closed until status is final
        312 # because status doesn't become final as soon as stream closes
        313 status = self.status()


    File /usr/lib/python3.9/concurrent/futures/_base.py:441, in Future.result(self, timeout)
        438 elif self._state == FINISHED:
        439     return self.__get_result()
    --> 441 self._condition.wait(timeout)
        443 if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
        444     raise CancelledError()


    File /usr/lib/python3.9/threading.py:312, in Condition.wait(self, timeout)
        310 try:    # restore state no matter what (e.g., KeyboardInterrupt)
        311     if timeout is None:
    --> 312         waiter.acquire()
        313         gotit = True
        314     else:


    KeyboardInterrupt: 


### Estimator

In Qiskit Terra, the Estimator base class establishes a uniform method for user interaction with all Estimator implementations. This facilitates the process of switching between simulators or devices used for calculating expectation values, even when the underlying implementation varies.


```python
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
observable = SparsePauliOp("ZZ")

estimator = Estimator()
job = estimator.run(circuit, observable)
print(f">>> Job ID: {job.job_id()}")
print(f">>> Job Status: {job.status()}")
result = job.result()
result
```

    >>> Job ID: 56dfd881-cb98-40a4-a157-9ec94561ac1f
    >>> Job Status: JobStatus.RUNNING





    EstimatorResult(values=array([1.]), metadata=[{}])



We can submit multiple circuit with:


```python
observable1 = SparsePauliOp("ZZ")
observable2 = SparsePauliOp("YY")
job = estimator.run([circuit] * 2, [observable1, observable2])
job.result()
```




    EstimatorResult(values=array([ 1., -1.]), metadata=[{}, {}])



### Error suppression, mitigation, correction

The 'Options' object enables the use of error suppression, mitigation, and correction without any issues.


```python
options = Options()
options.execution.shots = 1000
options.optimization_level = 0  # no optimization
options.resilience_level = 2  # ZNE, Zero noise extrapolation
```

