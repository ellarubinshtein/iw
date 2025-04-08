import cudaq
from typing import List
import time
import argparse
import random

# cudaq.set_target('qpp-cpu')
# cudaq.set_target('nvidia', option='mgpu') # GPU backend which enables scaling to large problem sizes
cudaq.set_target('nvidia')

# Record the start time
#start_time = time.time()

parser = argparse.ArgumentParser(description="Run the Deutsch-Jozsa algorithm on a quantum computer.")
parser.add_argument("qubit_count", type=int, help="Number of qubits for the algorithm (excluding the auxiliary qubit).")
args = parser.parse_args()

qubit_count = args.qubit_count

secret_string = [random.randint(0, 1) for _ in range(qubit_count)]

assert qubit_count == len(secret_string)


@cudaq.kernel
def oracle(register: cudaq.qview, auxiliary_qubit: cudaq.qubit,
           secret_string: List[int]):

    for index, bit in enumerate(secret_string):
        if bit == 1:
            x.ctrl(register[index], auxiliary_qubit)


@cudaq.kernel
def bernstein_vazirani(secret_string: List[int]):

    qubits = cudaq.qvector(len(secret_string))  # register of size n
    auxiliary_qubit = cudaq.qubit()  # auxiliary qubit

    # Prepare the auxillary qubit.
    x(auxiliary_qubit)
    h(auxiliary_qubit)

    # Place the rest of the register in a superposition state.
    h(qubits)

    # Query the oracle.
    oracle(qubits, auxiliary_qubit, secret_string)

    # Apply another set of Hadamards to the register.
    h(qubits)

    mz(qubits)  # measures only the main register


print(cudaq.draw(bernstein_vazirani, secret_string))
result = cudaq.sample(bernstein_vazirani, secret_string)

print(f"secret bitstring = {secret_string}")
print(f"measured state = {result.most_probable()}")
print(
    f"Were we successful? {''.join([str(i) for i in secret_string]) == result.most_probable()}"
)

