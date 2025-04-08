# Import the necessary libraries
import openfermion
import openfermionpyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator

import os
import timeit

import cudaq
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np


# Specify a molecular geometry, basis set, charge, and multiplicity.
geometry = [('O', (0.1173, 0.0, 0.0)), ('H', (-0.4691, 0.7570, 0.0)),
            ('H', (-0.4691, -0.7570, 0.0))]
basis = 'sto3g'
multiplicity = 1
charge = 0

# Classical preprocessing: 
# compute the Hartree Fock reference state and compute the integrals required for the Hamiltonian
molecule = openfermionpyscf.run_pyscf(
    openfermion.MolecularData(geometry, basis, multiplicity, charge))

# Building the Hamiltonian
molecular_hamiltonian = molecule.get_molecular_hamiltonian()

# The Hamiltonian must then be converted to a qubit Hamiltonian consisting of qubit operators.
fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)

# Jordan-Wigner transformation
qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

# Jordan-Wigner qubit Hamiltonian is converted into a CUDA-Q spin operator which 
# can be used to evaluate an expectation value given a quantum circuit.
spin_ham = cudaq.SpinOperator(qubit_hamiltonian)

# Quantum kernel to model the wavefunction
electron_count = 10
qubit_count = 2 * 7


@cudaq.kernel
def kernel(qubit_num: int, electron_num: int, thetas: list[float]):
    qubits = cudaq.qvector(qubit_num)

    for i in range(electron_num):
        x(qubits[i])

    cudaq.kernels.uccsd(qubits, thetas, electron_num, qubit_num)


parameter_count = cudaq.kernels.uccsd_num_parameters(electron_count,
                                                     qubit_count)

print(parameter_count)

# The cost function is defined as the expectation value of the Hamiltonian 
def cost(theta):

    exp_val = cudaq.observe(kernel, spin_ham, qubit_count, electron_count,
                            theta).expectation()

    return exp_val


exp_vals = []


def callback(xk):
    exp_vals.append(cost(xk))


# Initial variational parameters.
np.random.seed(42)
x0 = np.random.normal(0, 1, parameter_count)

# The VQE algorithm is run using the COBYLA optimizer.
cudaq.set_target('nvidia')
start_time = timeit.default_timer()
result = minimize(cost,
                  x0,
                  method='COBYLA',
                  callback=callback,
                  options={'maxiter': 50})
end_time = timeit.default_timer()

print('UCCSD-VQE energy =  ', result.fun)
print('Total number of qubits = ', qubit_count)
print('Total number of parameters = ', parameter_count)
print('Total number of terms in the spin hamiltonian = ',
      spin_ham.get_term_count())
print('Total elapsed time (s) = ', end_time - start_time)

plt.plot(exp_vals)
plt.xlabel('Epochs')
plt.ylabel('Energy')
plt.title('VQE')
plt.show()