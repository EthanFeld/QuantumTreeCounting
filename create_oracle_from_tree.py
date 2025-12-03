import numpy as np
from sklearn.tree import _tree

from qiskit import QuantumCircuit



def create_oracle_from_tree(
    tree,
    feature_names,
    all_feature_names,
    positive_class_index=1,  
):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    sorted_features = sorted(all_feature_names, key=lambda x: int(x.split("_")[1]))
    pixel_to_qubit = {name: i for i, name in enumerate(sorted_features)}
    num_vars = len(sorted_features)
    unique_condition_sets = set()
    #Recurse through tree to isolate branches for oracle
    def recurse(node, path_conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            if name == "undefined!":
                recurse(tree_.children_left[node], path_conditions)
                recurse(tree_.children_right[node], path_conditions)
                return
            recurse(tree_.children_left[node], path_conditions + [(name, 0)])
            recurse(tree_.children_right[node], path_conditions + [(name, 1)])
        else:
            class_idx = np.argmax(tree_.value[node])
            if class_idx == positive_class_index:
                key = tuple(sorted(path_conditions, key=lambda x: x[0]))
                unique_condition_sets.add(key)

    recurse(0, [])
    qc = QuantumCircuit(num_vars + 1)
    target_qubit = num_vars
    #Create Oracle from instructions above
    for cond_tuple in unique_condition_sets:
        control_indices = []
        flip_indices = []
        for pixel_name, required_state in cond_tuple:
            if pixel_name not in pixel_to_qubit:
                continue
            qubit_idx = pixel_to_qubit[pixel_name]
            control_indices.append(qubit_idx)
            if required_state == 0:
                flip_indices.append(qubit_idx)
        if not control_indices:
            continue
        if flip_indices:
            qc.x(flip_indices)
        qc.mcx(control_indices, target_qubit)
        if flip_indices:
            qc.x(flip_indices)
    return qc

