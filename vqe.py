import cudaq
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import argparse

# Single precision
cudaq.set_target("nvidia")
# Double precision
#cudaq.set_target("nvidia-fp64")

# Parse num of qubits from command line
parser = argparse.ArgumentParser(description="Run the Deutsch-Jozsa algorithm on a quantum computer.")
parser.add_argument("qubits", type=int, help="Number of qubits for the algorithm (excluding the auxiliary qubit).")
args = parser.parse_args()

# Number of hydrogen atoms.
hydrogen_count = args.qubits

# Distance between the atoms in Angstroms.
bond_distance = 0.7474

# Define a linear chain of Hydrogen atoms
geometry = [('H', (0, 0, i * bond_distance)) for i in range(hydrogen_count)]

spin = 1 if hydrogen_count % 2 != 0 else 0
hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
    geometry, 'sto-3g', 1, spin)

electron_count = data.n_electrons
qubit_count = 2 * data.n_orbitals

@cudaq.kernel
def kernel(thetas: list[float]):

    qubits = cudaq.qvector(qubit_count)

    for i in range(electron_count):
        x(qubits[i])

    cudaq.kernels.uccsd(qubits, thetas, electron_count, qubit_count)


parameter_count = cudaq.kernels.uccsd_num_parameters(electron_count,
                                                     qubit_count)

optimizer = cudaq.optimizers.COBYLA()

energy, parameters = cudaq.vqe(kernel,
                               hamiltonian,
                               optimizer,
                               parameter_count=parameter_count)

print(energy)
print("A100")