# Import the CUDA-Q package and set the target to run on NVIDIA GPUs.

import cudaq
import random
import time
import argparse
from typing import List

# Record the start time
# start_time = time.time()

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run the Deutsch-Jozsa algorithm on a quantum computer.")
parser.add_argument("qubits", type=int, help="Number of qubits for the algorithm (excluding the auxiliary qubit).")
args = parser.parse_args()

cudaq.set_target("nvidia")

# Number of qubits for the Deutsch-Jozsa algorithm, the last qubit is an auxiliary qubit
qubit_count = args.qubits
# Set the function to be "constant" or "balanced"
function_type = 'constant'

# Initialize fx depending on whether the function is constant or balanced
if function_type == 'constant':
    # For a constant function, fx is either all 0's or all 1's
    oracleType = 0  # 0 for constant
    fx_value = random.choice([0, 1])  # Randomly pick 0 or 1
    oracleValue = fx_value  # In constant case, fx_value is passed, for balanced it's not used
    fx = [fx_value] * (qubit_count - 1)
else:
    # For a balanced function, half of fx will be 0's and half will be 1's
    oracleType = 1
    fx = [0] * ((qubit_count - 1) // 2) + [1] * ((qubit_count - 1) - (qubit_count - 1) // 2)
    random.shuffle(fx)  # Shuffle to randomize the positions of 0's and 1's

# If needed initialize fx, oracleType, and oracleValue manually
#oracleType = 0
#oracleValue = 0
#fx = [0,0]

print(f"Generated fx for function type = {function_type}: {fx}")
print ("oracleType = ", oracleType)
print ("oracleValue = ", oracleValue)

# Define kernel
@cudaq.kernel
def kernel(fx: List[int], qubit_count: int, oracleType: int, oracleValue: int):
    
    # Allocate two input qubits
    input_qubits = cudaq.qvector(qubit_count-1)
    # Allocate an auxiliary qubit (initially |0⟩)
    auxiliary_qubit = cudaq.qubit()

    # Prepare the auxiliary qubit
    x(auxiliary_qubit)
    h(auxiliary_qubit)

    # Place the rest of the register in a superposition state
    h(input_qubits)

    # Logic for oracleType == 0 (constant oracle)
    if oracleType == 0:
        if oracleValue == 1:
            # Apply X gate to the auxiliary qubit
            x(auxiliary_qubit)
        elif oracleValue == 0:
            # Apply identity gate (do nothing)
            pass

    # Logic for oracleType == 1 (balanced oracle)
    elif oracleType == 1:
        for i in range(len(fx)):
            if fx[i] == 1:
                x.ctrl(input_qubits[i], auxiliary_qubit)

    # Apply Hadamard to the input qubit again after querying the oracle
    h(input_qubits)

    # Measure the input qubit to yield if the function is constant or balanced.
    mz(input_qubits)

print(cudaq.draw(kernel, fx, qubit_count, oracleType, oracleValue))

result = cudaq.sample(kernel, fx, qubit_count, oracleType, oracleValue, shots_count=1)

# Debugging: Print the raw result dictionary
print(f"Input qubits measurement outcome and frequency = {result}")

# Define the expected constant results for '00' and '11' for the number of input qubits
expected_constant_results = ['0' * (qubit_count - 1), '1' * (qubit_count - 1)]

# Check if either '00' or '11' (or their equivalent for more qubits) appears in the result
is_constant = any(result_key in result for result_key in expected_constant_results)


if is_constant:
    print("The oracle function is constant.")
else:
    print("The oracle function is balanced.")


print("GPU: V100")