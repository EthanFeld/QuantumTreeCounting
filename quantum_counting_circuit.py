from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
import numpy as np
from qiskit_aer import AerSimulator
def quantum_counting_circuit(oracle, num_precision_qubits, state_prep_circuit):
    # 1. Setup Registers
    num_state_qubits = state_prep_circuit.num_qubits
    num_oracle_qubits = oracle.num_qubits 
    
    qr_precision = QuantumRegister(num_precision_qubits, 'prec')
    qr_state = QuantumRegister(num_state_qubits, 'state')
    qr_target = QuantumRegister(1, 'target')
    cr = ClassicalRegister(num_precision_qubits, 'meas')
    
    qc = QuantumCircuit(qr_precision, qr_state, qr_target, cr)

    # 2. PRE-CONVERT TO GATES (The Fix)
    # We convert the circuit to a Gate object immediately.
    # This prevents the "Instruction cannot be converted" error later.
    try:
        state_prep_gate = state_prep_circuit.to_gate()
        state_prep_gate.label = "State Prep"
        
        # Create the inverse gate
        inv_state_prep_gate = state_prep_circuit.inverse().to_gate()
        inv_state_prep_gate.label = "State Prep Dagger"
        
        oracle_gate = oracle.to_gate()
        oracle_gate.label = "Oracle"
    except Exception as e:
        # Fallback if the circuit contains barriers or non-unitary instructions
        # We strip them to ensure it converts cleanly
        print(f"Warning: trimming circuit for gate conversion. Error: {e}")
        temp_qc = transpile(state_prep_circuit, basis_gates=['u', 'cx', 'id'])
        state_prep_gate = temp_qc.to_gate()
        inv_state_prep_gate = temp_qc.inverse().to_gate()
        oracle_gate = oracle.to_gate()

    # 3. Initialize State: |psi> |->
    qc.append(state_prep_gate, qr_state)
    qc.x(qr_target)
    qc.h(qr_target)

    # 4. Define the Grover Iterate (G) manually
    diffuser = QuantumCircuit(num_oracle_qubits, name="Diffuser")
    
    # Uncompute State Prep 
    diffuser.h(num_oracle_qubits - 1) 
    diffuser.x(num_oracle_qubits - 1) 
    diffuser.append(inv_state_prep_gate, range(num_state_qubits)) 
    
    # Reflection about 
    diffuser.x(range(num_oracle_qubits))
    diffuser.h(num_oracle_qubits - 1)
    diffuser.mcx(list(range(num_oracle_qubits - 1)), num_oracle_qubits - 1)
    diffuser.h(num_oracle_qubits - 1)
    diffuser.x(range(num_oracle_qubits))
    
    # Re-apply State Prep (
    diffuser.append(state_prep_gate, range(num_state_qubits))
    diffuser.x(num_oracle_qubits - 1)
    diffuser.h(num_oracle_qubits - 1)
    
    # Convert Diffuser to Gate
    diffuser_gate = diffuser.to_gate()
    diffuser_gate.label = "Diffuser"
    
    # C. Combine into Grover Operator G
    grover_it = QuantumCircuit(num_oracle_qubits)
    grover_it.append(oracle_gate, range(num_oracle_qubits))
    grover_it.append(diffuser_gate, range(num_oracle_qubits))
    
    grover_gate = grover_it.to_gate()
    grover_gate.label = "Grover"

    # 5. Phase Estimation Loop
    qc.h(qr_precision)
    
    for j in range(num_precision_qubits):
        power = 2**j
        if power == 1:
            ctrl_gate = grover_gate.control(1)
            qc.append(ctrl_gate, [qr_precision[j]] + list(qr_state) + list(qr_target))
        else:
            ctrl_gate = grover_gate.control(1)
            # Standard simulation loop
            for _ in range(power):
                 qc.append(ctrl_gate, [qr_precision[j]] + list(qr_state) + list(qr_target))

    # Inverse QFT and Measure
    qc.append(QFT(num_precision_qubits, inverse=True), qr_precision)
    qc.measure(qr_precision, cr)
    
    # Run
    simulator = AerSimulator()
    t_qc = transpile(qc, simulator, optimization_level=1)
    result = simulator.run(t_qc, shots=4096).result()
    
    return result.get_counts()