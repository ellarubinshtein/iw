import cudaq
from cudaq import spin
import cudaq_solvers as solvers
import numpy as np
from scipy.optimize import minimize 

geometry=[('C', (0.0, 0.0, 0.0)),
            ('H', (0.6295, 0.6295, 0.6295)),
            ('H', (0.6295, -0.6295, 0.6295)),
            ('H', (-0.6295, 0.6295, 0.6295)),
            ('H', (-0.6295, -0.6295, 0.6295))]
molecule = solvers.create_molecule(geometry,
                                            'sto-3g',    #basis set
                                            0,           #charge
                                            0,           #multiplicity
                                            nele_cas=2,
                                            norb_cas=3,
                                            ccsd=True,
                                            casci=True,
                                            verbose=True)

numQubits = molecule.n_orbitals * 2
print("Number of qubits: ", numQubits)
numElectrons = molecule.n_electrons

# Extract operators
operators=solvers.get_operator_pool("uccsd",
                                            num_qubits=numQubits,
                                            num_electrons=numElectrons)
 
# Retrieve number of operators                                            
count=len(operators)
 
# Make a list of initial parameters
init_params=[0.05]*count
print(init_params)
 
# Make final operator pool form operators and parameters
op_pool_uccsd=[1j*coef*op for coef,op in zip(init_params, operators)]

@cudaq.kernel
def initState(q: cudaq.qview):
    for i in range(numElectrons):
        x(q[i])

energy, thetas, ops = solvers.adapt_vqe(initState,
                                                molecule.hamiltonian,
                                                op_pool_uccsd,
                                                optimizer=minimize,
                                                method='L-BFGS-B',
                                                jac='3-point',
                                                tol=1e-7)
print('Adapt-VQE energy: ', energy)
print('Optimum pool operators: ', [op.to_string(False) for op in ops])

