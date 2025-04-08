# Import Libraries
from math import gcd, log2, ceil
import numpy as np
import random
import cudaq
from cudaq import *
import fractions
import matplotlib.pyplot as plt
import contfrac

# Inverse quantum Fourier transform¶
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

# Quantum kernels for modular exponentiation --> choosing a=5
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


# Phase kernel
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

control_register_size = 3
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


# Determining the order from the measurement results of the phase kernel¶
def top_results(sample_results, zeros, threshold):
    """Function to output the non-zero results whose counts are above the given threshold
    Returns
    -------
        dict[str, int]: keys are bit-strings and values are the respective counts
    """
    results_dictionary = {k: v for k, v in sample_results.items()}
    if zeros in results_dictionary.keys():
        results_dictionary.pop(zeros)
    sorted_results = {
        k: v for k, v in sorted(
            results_dictionary.items(), key=lambda item: item[1], reverse=True)
    }
    top_key = next(iter(sorted_results))
    max_value = sorted_results[top_key]
    top_results_dictionary = {top_key: max_value}

    for key in sorted_results:
        if results_dictionary[key] > min(threshold, max_value):
            top_results_dictionary[key] = results_dictionary[key]
    return top_results_dictionary

def get_order_from_phase(phase, phase_nbits, a, N):
    """Uses continued fractions to find the order of a mod N
    Parameters
    ----------
    phase: int
        Integer result from the phase estimate of U|x> = ax mod N
    phase_nbits: int
        Number of qubits used to estimate the phase
    a: int
        For this demonstration a is either 4 or 5
    N: int
        For this demonstration N = 21
    Returns
    -------
    int: period of a mod N if found, otherwise returns None
    """

    assert phase_nbits > 0
    assert a > 0
    assert N > 0

    eigenphase = float(phase) / 2**(phase_nbits)

    f = fractions.Fraction.from_float(eigenphase).limit_denominator(N)

    if f.numerator == 1:
        return None
    eigenphase = float(f.numerator / f.denominator)
    print('eigenphase is ', eigenphase)
    coefficients_continued_fraction = list(
        contfrac.continued_fraction(eigenphase))

    convergents_continued_fraction = list(contfrac.convergents(eigenphase))
    print('convergent sequence of fractions for this eigenphase is',
          convergents_continued_fraction)
    for r in convergents_continued_fraction:
        print(
            'using the denominators of the fractions in the convergent sequence, testing order =',
            r[1])
        if a**r[1] % N == 1:
            print('Found order:', r[1])
            return (r[1])
    return None

def find_order_quantum(a, N):
    """The quantum algorithm to find the order of a mod N, when x = 4 or x =5 and N = 21
    Parameters
    ----------
    a: int
        For this demonstration a will be either 4 or 5
    N: int
        For this demonstration N will be 21
    Returns
    r: int the period if it is found, or None if no period is found
    -------

    """

    if (a == 4 and N == 21) or (a == 5 and N == 21):
        shots = 15000
        if a == 4 and N == 21:
            control_register_size = 3
            work_register_size = 2
        if a == 5 and N == 21:
            control_register_size = 5
            work_register_size = 5

        #cudaq.set_random_seed(123)
        results = cudaq.sample(phase_kernel,
                               control_register_size,
                               work_register_size,
                               a,
                               N,
                               shots_count=shots)
        print("Measurement results:")
        print(results)

        # We will want to ignore the all zero result
        zero_result = ''.join(
            [str(elem) for elem in [0] * control_register_size])
        # We'll only consider the top results from the sampling
        threshold = shots * (.1)
        most_probable_bitpatterns = top_results(results, zero_result, threshold)

        for key in most_probable_bitpatterns:
            # Convert the key bit string into an integer
            # This integer divided by 8 is an estimate for the phase
            reverse_result = key[::-1]
            phase = int(reverse_result, 2)

            print("Trying nonzero bitpattern from the phase estimation:", key,
                  "=", phase)
            r = get_order_from_phase(phase, control_register_size, a, N)
            if r == None:
                print('No period found.')

                continue

            return r
            break
    else:
        print(
            "A different quantum kernel is required for this choice of a and N."
        )

my_integer = 21
initial_value_to_start = 5  # Try replacing 5 with 4
quantum = True
shors_algorithm(my_integer, initial_value_to_start, quantum)