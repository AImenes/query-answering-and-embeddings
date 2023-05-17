import os
import pickle

# Custom modules
import perfectref_v1 as pr
from perfectref_v1 import Query, QueryBody
from perfectref_v1 import AtomParser, AtomConcept, AtomRole, AtomConstant
from perfectref_v1 import Variable, Constant

def query_structure_parsing(query_string, query_structure, tbox):
    """
    Parse a given query string based on its structure and the TBox ontology provided.
    
	Example: If we have a up query (with a disjunction) it will be restructured:
    q(w): P22(x,y) | P25(x,y) ^ P40(y, w) 
	->
	q1: P22(x,y) ^ P40(y, w) 
	q2: P25(x,y) ^ P40(y, w) 
	q = q1 union q2

    Args:
        query_string (str): The query string to be parsed.
        query_structure (str): The structure of the query ('up', '2u', or other).
        tbox (Ontology): The TBox ontology for the knowledge graph.

    Returns:
        dict: A dictionary containing the parsed query information, including the query structure and parsed query
              expressions.
    """
    # Remove angle brackets from the query string
    char_remove = ['<', '>']
    for char in char_remove:
        query_string = query_string.replace(char, '')

    # Handle up and 2u queries. That is, queries that contains disjunction.
    if query_structure == 'up' or query_structure == '2u':
        head, body = query_string.split(":-")

        # Check if the query contains a conjunction (^)
        if "^" in body:
            the_or_atoms, last_atom = body.split("^")
            first_atom, second_atom = the_or_atoms.split(" | ")
            q1 = head + ":-" + first_atom + "^" + last_atom
            q2 = head + ":-" + second_atom + "^" + last_atom
        else:
            first_atom, second_atom = body.split(" | ")
            q1 = head + ":-" + first_atom
            q2 = head + ":-" + second_atom

        return {
            'query_structure': query_structure,
            'q1': get_name_and_namespace(parse_query(q1, query_structure), tbox),
            'q2': get_name_and_namespace(parse_query(q2, query_structure), tbox)
        }
    # Handle other query structures. That is projection and intersection queries.
    else:
        return {
            'query_structure': query_structure,
            'q1': get_name_and_namespace(parse_query(query_string, query_structure), tbox),
            'q2': None
        }

def parse_query(query_string: str, query_structure: str = None) -> Query:
    """
    Parse a query string into a Query object.

    This function takes a query string and a query structure (optional) and parses it into a Query object.

    Args:
        query_string (str): The query string to parse.
        query_structure (str, optional): The structure of the query. Defaults to None.

    Returns:
        Query: A Query object representing the parsed query string.
    """

    # Split the query string into head and body
    head_string, body_string = query_string.split(":-")
    dictionary_of_variables = {}

    # Remove trailing spaces
    head_string = head_string.strip()
    body_string = body_string.strip()

    # Parse the head and body of the query string
    head = parse_head(head_string, dictionary_of_variables)
    body = parse_body(body_string, dictionary_of_variables)

    # Update the boundness Boolean variables after the recursion
    initial_update_entries(head, dictionary_of_variables)
    [initial_update_entries(b, dictionary_of_variables) for b in body]

    # Update classtypes
    new_body = list()
    for atom in body:
        if atom.type == "CONSTANT":
            # Create a new AtomConstant object
            new_body.append(AtomConstant(None, atom.get_value(), atom.name))
            
        elif atom.type == "CONCEPT":
            # Create a new AtomConcept object
            new_body.append(AtomConcept(None, atom.var1, atom.name))
            
        elif atom.type == "ROLE":
            # Create a new AtomRole object
            new_body.append(AtomRole(None, atom.var1, atom.var2, False, atom.name))
            
        else:
            # Print an error message for syntax errors
            raise SyntaxError("SYNTAX ERROR")

    # Return a Query object with the parsed query string and query structure
    return Query(head, QueryBody(new_body), dictionary_of_variables, query_structure)

def parse_head(head_string: str, dictionary_of_variables: dict) -> AtomParser:
    """
    Parse a head string into an Atom object.

    This function takes a head string and a dictionary of variables and parses it into an Atom object.

    Args:
        head_string (str): The head string to parse.
        dictionary_of_variables (DictOfVariables): A dictionary containing information about the variables in the query.

    Returns:
        AtomParser: An Atom object representing the parsed head string.
    """

    # Set the distinguished flag to True
    is_distinguished = True 

    # Parse the head string into an Atom object
    return parse_atom(head_string, is_distinguished, dictionary_of_variables)
 
def parse_body(body_string: str, dictionary_of_variables: dict) -> list:
    """
    Parse a body string into a list of Atom objects.

    This function takes a body string and a dictionary of variables and parses it into a list of Atom objects.

    Args:
        body_string (str): The body string to parse.
        dictionary_of_variables (dict): A dictionary containing information about the variables in the query.

    Returns:
        list: A list of AtomParser objects representing the parsed body string.
    """

    # Set the distinguished flag to False
    is_distinguished = False

    # Split the body string into atom strings
    atom_str_list = body_string.split("^")

    # Parse each atom in the atom string list
    return [parse_atom(atom_str, is_distinguished, dictionary_of_variables) for atom_str in atom_str_list]

def parse_atom(atom_string: str, is_distinguished: bool, dictionary_of_variables: dict) -> AtomParser:
    """
    Parse an atom string into an Atom object.

    This function takes an atom string, a flag indicating whether the variable is distinguished, and a dictionary of
    variables, and parses it into an Atom object.

    Args:
        atom_string (str): The atom string to parse.
        is_distinguished (bool): A flag indicating whether the variable is distinguished.
        dictionary_of_variables (DictOfVariables): A dictionary containing information about the variables in the query.

    Returns:
        AtomParser: An AtomParser object representing the parsed atom string. Will later we used to split to ROLES and CONCEPTS.
    """

    # Extract tokens from the atom string
    iri, entry_str_list = extract_entry_tokens(atom_string)

    # Parse each entry in the entry string list
    entry_list = [parse_entry(token, is_distinguished, dictionary_of_variables) for token in entry_str_list]

    # Create a new AtomParser object with the IRI and entry list
    return AtomParser(iri, entry_list)

def parse_entry(entry_string: str, is_distinguished: bool, dictionary_of_variables: dict) -> Variable or Constant:
    """
    Parse an entry string into a Variable or Constant object based on its prefix.

    This function takes an entry string and parses it into a Variable or Constant object based on whether the string
    starts with "?". If it is a variable, it also updates the dictionary of variables with the new variable.

    Args:
        entry_string (str): The string to be parsed.
        is_distinguished (bool): A flag indicating whether the variable is the distinguished variable in the query.
        dictionary_of_variables (dict): A dictionary containing the variables in the query and their indices.

    Returns:
        Variable or Constant: A Variable or Constant object based on the prefix of the entry string.
    """

    # Check if the entry string is a variable
    if entry_string.startswith("?"):
        # If it is, create a new Variable object and update the dictionary of variables
        variable = Variable(entry_string, parse_dict_of_variables(entry_string, is_distinguished, dictionary_of_variables))
        return variable
    
    # If it's not a variable, create a new Constant object
    return Constant(entry_string)

def extract_entry_tokens(atom_string: str) -> tuple:
    """
    Extract tokens from an atom string and return them as a tuple.

    This function takes an atom string and extracts the entries and IRI from it. It returns them as a tuple.

    Args:
        atom_string (str): The atom string to extract tokens from.

    Returns:
        tuple: A tuple containing the IRI and a list of entries extracted from the atom string.
    """

    entries_list = list()
    iri = ""

    # Check if the atom string is a constant
    if (not ("(") in atom_string):
        iri = atom_string

    # Check if the atom string is a constant with empty parentheses
    elif (("()") in atom_string):
        iri = atom_string.split(("("))[0]

    else:
        # If the atom string contains entries
        iri, entries = atom_string.split("(", 1)
        entries = entries.replace(" ", "")
        entries = entries.rstrip(")")

        if not ((",") in entries):
	    
            # If there is only one entry, add it to the list
            entries_list.append(entries)
        else:
            # If there are multiple entries, split them and add them to the list
            entries = entries.split(",")
            for e in entries:
                entries_list.append(e)

    return iri, entries_list

def parse_dict_of_variables(entry_string, is_distinguished, dictionary_of_variables):
    """
    Update the dictionary_of_variables for a given entry_string based on whether it is distinguished or not.
    
    Args:
        entry_string (str): The variable entry string to be processed
        is_distinguished (bool): A flag indicating whether the entry_string is a distinguished variable
        dictionary_of_variables (dict): A dictionary containing information about variables

    Returns:
        dict: An updated variable entry from the dictionary_of_variables
    """
    
    # Check if the variable is already in the dictionary_of_variables
    if entry_string in dictionary_of_variables:
        # If the variable is in the body, mark it as shared
        if dictionary_of_variables[entry_string]['in_body']:
            dictionary_of_variables[entry_string]['is_shared'] = True
        else:
            dictionary_of_variables[entry_string]['in_body'] = True
    else:
        # Create a new variable entry based on the is_distinguished flag
        new_entry = {'is_bound': False, 'is_distinguished': is_distinguished, 'in_body': True, 'is_shared': False}
        dictionary_of_variables[entry_string] = new_entry

    # Update the 'is_bound' attribute if the variable is shared or distinguished
    if dictionary_of_variables[entry_string]['is_shared'] or dictionary_of_variables[entry_string]['is_distinguished']:
        dictionary_of_variables[entry_string]['is_bound'] = True

    return dictionary_of_variables[entry_string]

def initial_update_entries(atom, dict_of_variables):
	
	"""
    Update entries in an atom with values from the dictionary of variables.

    This function takes an atom and a dictionary of variables and updates each entry in the atom with its corresponding
    values from the dictionary.

    Args:
        atom (Atom): The atom to update.
        dict_of_variables (dict): A dictionary containing information about the variables in the query.
    """

	for entry in atom.get_entries():
		e = dict_of_variables[entry.original_entry_name]
		entry.update_values(e['is_distinguished'], e['in_body'], e['is_shared'])

def update_processed_status(current_query, PR, status):
    """
    
    This method updates the status on a query if it has been used for rewriting, to avoid a never ending rewriting loop.

    Args:
        current_query (QueryBody): The QueryBody object to update.
        PR (dict): PerfectRef results dict
        status (bool): Flag for whether atom is processed

    Returns:
        Nothing. Changes objects.

    """
    for query in PR:
        if current_query == query:
            query.set_process_status(status)

def update_body(body: QueryBody) -> QueryBody:
    """
    Update the variables in a QueryBody object.

    This function takes a QueryBody object, updates the variables in each Atom object in the QueryBody object, and returns
    the updated QueryBody object.

    Args:
        body (QueryBody): The QueryBody object to update.

    Returns:
        QueryBody: The updated QueryBody object.
    """

    # Create a dictionary of variables
    dictionary_of_variables = {}

    # Update each Atom object in the QueryBody object
    for atom in body.body:
        update_atom(atom, dictionary_of_variables)

    # Update the boundness Boolean variables after the recursion
    [initial_update_entries(b, dictionary_of_variables) for b in body.body]

    # Return the updated QueryBody object
    return body

def update_atom(atom, dictionary_of_variables):
    """
    Update an AtomConcept or AtomRole with the given dictionary of variables.

    Args:
        atom (AtomConcept or AtomRole): The AtomConcept or AtomRole to be updated.
        dictionary_of_variables (dict): A dictionary mapping variable names to their values.

    Returns:
        None
    """
    if isinstance(atom, AtomConcept):
        # If the atom is an AtomConcept, update its first variable.
        update_concept(atom.var1, dictionary_of_variables)
    else:
        # If the atom is an AtomRole, update its first and second variables.
        update_role(atom.var1, atom.var2, dictionary_of_variables)

def update_concept(var, dictionary_of_variables):
    """
    Update a concept with the given dictionary of variables.

    Args:
        var (AtomConcept): The consept variable to be updated.
        dictionary_of_variables (dict): A dictionary mapping variable names to their values.

    Returns:
        None
    """
    iri = var.original_entry_name
    # Parse the IRI of the variable and update its distinguished attribute.
    parse_dict_of_variables(iri, var.distinguished, dictionary_of_variables)

def update_role(var1, var2, dictionary_of_variables):
    """
    Update a role with the given dictionary of variables.

    Args:
        var1 (AtomRole): The first role variable to be updated.
        var2 (AtomRole): The second role variable to be updated.
        dictionary_of_variables (dict): A dictionary mapping variable names to their values.

    Returns:
        None
    """
    iri1, iri2 = var1.original_entry_name, var2.original_entry_name
    
    # Parse the IRIs of the variables and update their distinguished attributes.
    parse_dict_of_variables(iri1, var1.distinguished, dictionary_of_variables)
    parse_dict_of_variables(iri2, var2.distinguished, dictionary_of_variables)

def get_name_and_namespace(q, tbox):
    """
    Based on a QueryBody and a TBox, add the atom names to the object.

    """
    classes = list(tbox.classes())
    properties = list(tbox.properties())

    #for every atom in q
    for g in q.body.body:
            
        #Classes
        matches = list()
        counter = 0

        #for every class
        for cl in classes:

            #if atom name equals class name
            if g.iri == cl.iri:

                #save class
                matches.append(cl)

                #add to counter
                counter += 1

        #if counter is 1
        if counter == 1:

            #set namespace and iri
            #g.set_namespace(matches[0].namespace.ontology)
            g.set_name(matches[0].name)


        #Properties
        matches = list()
        counter = 0
        #for every property
        for pp in properties:

            #if atom name equals class name
            if g.iri == pp.iri:

                #save class
                matches.append(pp)

                #add to counter
                counter += 1

        #if counter is 1
        if counter == 1:
            #set namespace and iri
            #g.set_namespace(matches[0].namespace.ontology)
            g.set_name(matches[0].name)
    return q

def query_reformulate(parsed_generated_queries: dict, rewriting_upper_limit: int, full_pth: str, t_box_path: str) -> dict:
    """
    Perform query reformulation using PerfectRef and return the reformulated queries.

    This function takes a dictionary of parsed queries, an upper limit for rewriting, a path to a file to save the reformulated queries,
    and a path to a T-Box file, and performs query reformulation using PerfectRef on each query in the parsed query dictionary. If the
    reformulated queries have not been saved to the file specified by full_pth, the function performs the reformulation and saves the
    result to the file. If the file already exists, the function loads the reformulated queries from the file.

    Args:
        parsed_generated_queries (Dict[str, Any]): A dictionary of parsed queries to reformulate.
        rewriting_upper_limit (int): An upper limit for query rewriting.
        full_pth (str): The path to the file to save the reformulated queries.
        t_box_path (str): The path to the T-Box file.

    Returns:
        dict: A dictionary of the reformulated queries.
    """

    # If the reformulated queries have not been saved to the file specified by full_pth
    if not os.path.exists(full_pth):
        # For each query structure
        for query_structure in parsed_generated_queries.keys():
            print("Performing PerfectRef rewriting for structure " + query_structure + "...")
            # For each query in that structure
            for query_dict in parsed_generated_queries[query_structure]:
                # If the query structure is not up or 2u
                if not (query_structure == 'up' or query_structure == '2u'):
                    # Perform PerfectRef
                    query_dict['rewritings'] = pr.get_entailed_queries(t_box_path, query_dict['q1'], upperlimit=rewriting_upper_limit, parse=False)
                else:
                    temp1 = pr.get_entailed_queries(t_box_path, query_dict['q1'], upperlimit=rewriting_upper_limit, parse=False)
                    temp2 = pr.get_entailed_queries(t_box_path, query_dict['q2'], upperlimit=rewriting_upper_limit, parse=False)
                    query_dict['rewritings'] = temp1 + temp2

    # else:
    #     # If the reformulated queries have already been saved to the file specified by full_pth, load them from the file
    #     print("\n Reformulation already exists. Loaded pickle for this configuration. Delete or rename the pickle file if you want to redo the reformulation. \n")
    #     with open(full_pth, 'rb') as handle:
    #         parsed_generated_queries = pickle.load(handle)

    # Return the reformulated queries
    return parsed_generated_queries
