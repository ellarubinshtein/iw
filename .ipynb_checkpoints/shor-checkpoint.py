import cudaq
import argparse

# Define kernels for the quantum Fourier transform and the inverse quantum Fourier transform
@cudaq.kernel
def quantum_fourier_transform(qubits: cudaq.qview):
    qubit_count = len(qubits)
    # Apply Hadamard gates and controlled rotation gates.
    for i in range(qubit_count):
        h(qubits[i])
        for j in range(i + 1, qubit_count):
            angle = (2 * np.pi) / (2**(j - i + 1))
            cr1(angle, [qubits[j]], qubits[i])


@cudaq.kernel
def inverse_qft(qubits: cudaq.qview):
    cudaq.adjoint(quantum_fourier_transform, qubits)


@cudaq.kernel
def modular_mult_5_21(work: cudaq.qview):
    """"Kernel for multiplying by 5 mod 21
    based off of the circuit diagram in
    https://physlab.org/wp-content/uploads/2023/05/Shor_s_Algorithm_23100113_Fin.pdf
    Modifications were made to change the ordering of the qubits"""
    x(work[0])
    x(work[2])
    x(work[4])

    swap(work[0], work[4])
    swap(work[0], work[2])


@cudaq.kernel
def modular_exp_5_21(exponent: cudaq.qview, work: cudaq.qview,
                     control_size: int):
    """ Controlled modular exponentiation kernel used in Shor's algorithm
    |x> U^x |y> = |x> |(5^x)y mod 21>
    """
    x(work[0])
    for exp in range(control_size):
        ctrl_qubit = exponent[exp]
        for _ in range(2**(exp)):
            cudaq.control(modular_mult_5_21, ctrl_qubit, work)

# Demonstrate iterated application of 5y mod 21 where y = 1
@cudaq.kernel
def demonstrate_mod_exponentiation(iterations: int):
    qubits = cudaq.qvector(5)
    x(qubits[0]
     )  # initalizes the qubits in the state for y = 1 which is |10000>
    for _ in range(iterations):
        modular_mult_5_21(qubits)


@cudaq.kernel
def modular_exp_4_21(exponent: cudaq.qview, work: cudaq.qview):
    """ Controlled modular exponentiation kernel used in Shor's algorithm
     |x> U^x |y> = |x> |(4^x)y mod 21>
     based off of the circuit diagram in https://arxiv.org/abs/2103.13855
     Modifications were made to account for qubit ordering differences"""
    swap(exponent[0], exponent[2])
    # x = 1
    x.ctrl(exponent[2], work[1])

    # x = 2
    x.ctrl(exponent[1], work[1])
    x.ctrl(work[1], work[0])
    x.ctrl([exponent[1], work[0]], work[1])
    x.ctrl(work[1], work[0])

    # x = 4
    x(work[1])
    x.ctrl([exponent[0], work[1]], work[0])
    x(work[1])
    x.ctrl(work[1], work[0])
    x.ctrl([exponent[0], work[0]], work[1])
    x.ctrl(work[1], work[0])
    swap(exponent[0], exponent[2])


@cudaq.kernel
def phase_kernel(control_register_size: int, work_register_size: int, a: int,
                 N: int):
    """
    Kernel to estimate the phase of the modular multiplication gate |x> U |y> = |x> |a*y mod 21> for a = 4 or 5
    """

    qubits = cudaq.qvector(control_register_size + work_register_size)
    control_register = qubits[0:control_register_size]
    work_register = qubits[control_register_size:control_register_size +
                           work_register_size]

    h(control_register)

    if a == 4 and N == 21:
        modular_exp_4_21(control_register, work_register)
    if a == 5 and N == 21:
        modular_exp_5_21(control_register, work_register, control_register_size)

    inverse_qft(control_register)

    # Measure only the control_register and not the work_register
    mz(control_register)

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run the Deutsch-Jozsa algorithm on a quantum computer.")
parser.add_argument("qubits", type=int, help="Number of qubits for the algorithm (excluding the auxiliary qubit).")
args = parser.parse_args()

control_register_size = args.qubits # changing this for experiment part
work_register_size = 5
values_for_a = [4, 5]
idx = 1  # change to 1 to select 5
N = 21
shots = 15000

print(
    cudaq.draw(phase_kernel, control_register_size, work_register_size,
               values_for_a[idx], N))

results = cudaq.sample(phase_kernel,
                       control_register_size,
                       work_register_size,
                       values_for_a[idx],
                       N,
                       shots_count=shots)
print(
    "Measurement results for a={} and N={} with {} qubits in the control register "
    .format(values_for_a[idx], N, control_register_size))
print(results)