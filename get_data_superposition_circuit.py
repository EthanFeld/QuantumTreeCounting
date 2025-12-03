import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

def get_data_superposition_circuit(data, feature_indices=None):

    num_samples = len(data)
    sample_length = len(data[0])
    if feature_indices is None:
        feature_indices = list(range(sample_length))
    
    num_features = len(feature_indices)
    if num_features > 25:
        raise ValueError(f"Too many features ({num_features}); max ~25 for simulation.")
    
    state_vector_size = 2**num_features
    state_vector = np.zeros(state_vector_size, dtype=complex)
    #encode data
    for sample in data:
        sample_index = 0
        for i, orig_idx in enumerate(feature_indices):
            value = sample[orig_idx]
            bit_val = 1 if value > 0.5 else 0
            shift = i  
            sample_index |= (bit_val << shift)
        state_vector[sample_index] = 1.0
    #Find amplitudes and normalize
    state_vector = np.sqrt(state_vector)
    norm = np.linalg.norm(state_vector)
    if norm == 0:
        state_vector[0] = 1.0
    else:
        state_vector /= norm
    
    qc = QuantumCircuit(num_features)
    state_prep = StatePreparation(state_vector)
    qc.append(state_prep, range(num_features))
    return qc
