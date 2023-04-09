import os
import pickle
from torch.cuda import is_available
from owlready2 import get_ontology

#custom modules
from kglookup import *

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def press_any_key():
    input("\n\nPress any key to continue ...")

def get_device():
    """
    This method checks whether a GPU is available, and returns the
    appropriate device string to use in the rest of the program.

    Returns:
        str: The device string, either 'gpu' or 'cpu', based on availability.
    """
    # Check if GPU is available for computation, if not return CPU
    return 'gpu' if is_available() else 'cpu'

def load_ontology(path):
    return get_ontology(path).load()

def create_test_environment(project_name: str) -> None:
    """
    Create a test environment for a given project.

    This function creates a new directory for the test environment if it doesn't already exist. If the directory
    already exists, the user is informed and no action is taken.

    Args:
        project_name (str): Name of the project for which the test environment is to be created.

    Returns:
        None
    """

    # Clear the console
    clear()

    # Construct the test environment directory path
    env_path = f"testcases/{project_name}"

    # Check if the directory already exists
    if os.path.isdir(env_path):
        # If it exists, inform the user and avoid overwriting the existing test environment
        print("This test case already exists. You will not overwrite a model configuration, i.e, if the same model, "
              "dimension and epoch number.\n If you want a new test environment, please change the identifier.")
        press_any_key()
    else:
        # If it doesn't exist, create the directory and inform the user
        os.makedirs(env_path)
        print("Test environment successfully created.")
        press_any_key()

def update_prediction_pickle(structure, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(structure, file, protocol=pickle.HIGHEST_PROTOCOL)

def build_variable_hierarchy(query, distinguished_variable, processed_vars=None, depth=0):
    """
    Create a variable hierarchy in a query based on the distinguished variable.

    Args:
        query (QueryBody):              The input query with a body containing atoms. 
        distinguished_variable (str):   The distinguished variable used as the starting point for building the hierarchy.
        processed_vars (set, optional): A set of variables that have been processed. Defaults to None.
        depth (int, optional):          The depth in the hierarchy for the current distinguished variable. Defaults to 0.

    Returns:
        dict: A dictionary representing the variable hierarchy, with variables as keys and their depth in the hierarchy as values.
    """

    # Initialize the processed_vars set if not provided
    if processed_vars is None:
        processed_vars = set()

    # Create the initial hierarchy with the distinguished_variable at the given depth
    hierarchy = {distinguished_variable: depth}

    # Add the distinguished_variable to the set of processed variables
    processed_vars.add(distinguished_variable)

    # Iterate through the atoms in the query body
    for atom in query.body:

        # If the atom is an AtomRole
        if isinstance(atom, AtomRole):

            # Extract the variable names from the atom
            var1, var2 = atom.var1.original_entry_name, atom.var2.original_entry_name

            # Check if either of the variables matches the distinguished_variable
            if distinguished_variable == var1 or distinguished_variable == var2:

                # Identify the child variable in the atom
                child = var1 if var2 == distinguished_variable else var2

                # If the child variable has not been processed yet
                if child not in processed_vars:

                    # Recursively build the hierarchy for the child variable and update the current hierarchy
                    hierarchy.update(build_variable_hierarchy(query, child, processed_vars, depth + 1))

        # If the atom is an AtomConcept
        elif isinstance(atom, AtomConcept):

            # Extract the variable name from the atom
            var1 = atom.var1.original_entry_name

            # If the variable has not been processed yet
            if var1 not in processed_vars:

                # Update the hierarchy with the new variable
                hierarchy.update(hierarchy)

    # Return the constructed variable hierarchy
    return hierarchy

def inverse_hierarchy(hierarchy):
    """
    Takes a hierarchy dictionary as input and returns the inverse of the dictionary
    where the depths are the keys.

    Args:
        hierarchy (dict): A dictionary representing the variable hierarchy, where
                         keys are variable names and values are the depths.

    Returns:
        dict: The inverse of the hierarchy dictionary, where keys are depths and
              values are lists of variable names.
    """
    inverse = {}
    for variable, depth in hierarchy.items():
        if depth not in inverse:
            inverse[depth] = []
        inverse[depth].append(variable)
    return inverse

def get_target_variable(hierarchy: dict, var1: str, var2: str) -> tuple:
    """
    Determine the target variable based on the hierarchy and the depths of the given variables.

    This function compares the depths of two variables in a given hierarchy. It returns the target variable
    along with the corresponding 'head' or 'tail' designation. If both variables have the same depth, it returns
    a tuple of None values.

    Args:
        hierarchy (dict): A dictionary representing the hierarchy, where keys are variable names and values are their depths.
        var1 (str): The first variable to compare.
        var2 (str): The second variable to compare.

    Returns:
        tuple: A tuple containing the designation ('head' or 'tail'), the target variable, and the other variable.
               If both variables have the same depth, a tuple of None values is returned.
    """

    # Check if both variables are present in the hierarchy
    if var1 not in hierarchy or var2 not in hierarchy:
        return None, None, None

    # Get the depths of both variables from the hierarchy
    depth1 = hierarchy[var1]
    depth2 = hierarchy[var2]

    # Compare the depths to determine the target variable
    if depth1 < depth2:
        return 'head', var1, var2
    elif depth1 > depth2:
        return 'tail', var2, var1
    else:
        # If both variables have the same depth, return a tuple of None values
        return None, None, None
    
def sort_atoms_by_depth(hierarchy: dict, query: QueryBody):
    """
    Sort atoms in the query body based on their variable depths in the given hierarchy.

    This function sorts atoms in the query body based on the depths of their variables within a given hierarchy.
    Atoms with higher variable depths are placed earlier in the sorted list.

    Args:
        hierarchy (dict): A dictionary representing the hierarchy, where keys are variable names and values are their depths.
        query (QueryBody): An object containing the query body, which is a list of atoms (AtomRole and AtomConcept).

    Returns:
        List[Union[AtomRole, AtomConcept]]: A sorted list of atoms in the query body.
    """
    
    # Create a dictionary to store the depth of each variable
    variable_depths = hierarchy

    # Define a custom sorting key function for sorting the atoms based on variable depth
    def atom_depth_key(atom):
        if isinstance(atom, AtomRole):
            # Get the depth of both variables in the atom
            var1, var2 = atom.var1.original_entry_name, atom.var2.original_entry_name
            depth1 = variable_depths[var1]
            depth2 = variable_depths[var2]
            # Return the maximum depth, which represents the higher variable in the hierarchy
            return max(depth1, depth2)
        if isinstance(atom, AtomConcept):
            return variable_depths[atom.var1.original_entry_name]
        
        return None

    # Sort the atoms based on the variable depth using the custom sorting key
    sorted_atoms = sorted(query.body, key=atom_depth_key, reverse=True)

    return sorted_atoms

def get_query_string(querybody: QueryBody, distinguished_variable: str) -> str:
    """
    Generate a query string based on a query body and a distinguished variable.

    This function generates a query string from a given query body and a distinguished variable. The query
    string can be used to query a knowledge base.

    Args:
        querybody (QueryBody): An object containing the query body, which is a list of atoms (AtomRole and AtomConcept).
        distinguished_variable (str): The distinguished variable in the query.

    Returns:
        str: A query string based on the query body and distinguished variable.
    """

    # Create the initial part of the query string with the distinguished variable
    temp = "q(" + distinguished_variable + ") :- "

    # Iterate over the atoms in the query body and add them to the query string
    for atom in querybody.body:
        if isinstance(atom, AtomConcept):
            # Add a concept atom to the query string
            temp += atom.name + "(" + atom.var1.original_entry_name + ")"
        if isinstance(atom, AtomRole):
            # Add a role atom to the query string
            temp += atom.name + "(" + atom.var1.original_entry_name + "," + atom.var2.original_entry_name + ")"
        
        # Add a separator between atoms
        temp += "^"

    # Remove the final separator and return the query string
    return temp[:-1]

