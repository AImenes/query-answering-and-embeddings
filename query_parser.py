import os
import pickle

import perfectref_v1 as pr
from perfectref_v1 import Query, QueryBody
from perfectref_v1 import AtomParser, AtomConcept, AtomRole, AtomConstant
from perfectref_v1 import Variable, Constant


"""
	METHODS FOR PARSING

		parse_query					Splits head and body, creates a dictionary for variables (which is within the object). Return type Class:Query

		parse_head					Since the head per definition contains distinguished variables, the boolean variable is_distinguised is set to True. 
 									This is passed to Atom-parser. It returns a Class:Atom, 

		parse_body					Since the head per definition contains distinguished variables, the boolean variable in the body is_distinguised is set to False. 
									This is passed to Atom-parser. It returns a list of Class:Atom.

		parse_atom					Extracts entry tokens from a Atom, and returns an Atom with a list of entries. Returns either 

		parse_entry					Parses each entry as either a Variable or Constant. Returns a Class:Variable or Class:Constant


	METHODS FOR UTILITIES

		extract_entry_tokens		Parses the entire atom, and extracts the variables or constants
	
		parse_dict_of_variables		Checks if variable already exists. That is, a shared (bound) variable. Returns Boolean value.

		update_entries				After the recursion is complete, all the entries are detected and decided whether its bound, shared or unbound. This updates those objects.

"""
## Query Structures
def query_structure_parsing(query_string, query_structure, tbox):
	char_remove = ['<', '>']
	for char in char_remove:
		query_string = query_string.replace(char,'')
	if query_structure == 'up' or query_structure == '2u':
		head, body = query_string.split(":-")
		if "^" in body:
			the_or_atoms, last_atom = body.split("^")
			first_atom, second_atom = the_or_atoms.split(" | ")
			q1 = head + ":-" + first_atom + "^" + last_atom
			q2 = head + ":-" + second_atom + "^" + last_atom
		else:
			first_atom, second_atom = body.split(" | ")
			q1 = head + ":-" + first_atom
			q2 = head + ":-" + second_atom
        
		return {'query_structure': query_structure, 'q1': get_name_and_namespace(parse_query(q1, query_structure), tbox), 'q2': get_name_and_namespace(parse_query(q2, query_structure), tbox)}
	else:
		return {'query_structure': query_structure, 'q1': get_name_and_namespace(parse_query(query_string, query_structure), tbox), 'q2': None}


## PARSING
def parse_query(query_string, query_structure = None):    

    # Split head and body
	head_string, body_string = query_string.split(":-")
	dictionary_of_variables = {}

	# Remove trailing spaces
	head_string = head_string.strip()
	body_string = body_string.strip()

	# Initial recursion for atom detection and variable detection
	head = parse_head(head_string, dictionary_of_variables)
	body = parse_body(body_string, dictionary_of_variables)

	#Update the boundness Boolean variables after the recursion
	initial_update_entries(head, dictionary_of_variables)
	[initial_update_entries(b, dictionary_of_variables) for b in body]

	#Update classtypes
	new_body = list()
	for atom in body:
		if atom.get_type() == "CONSTANT":
			new_body.append(AtomConstant(None,atom.get_value(),atom.get_name()))
			
		elif atom.get_type() == "CONCEPT":
			new_body.append(AtomConcept(None,atom.get_var1(),atom.get_name()))
			
		elif atom.get_type() == "ROLE":
			new_body.append(AtomRole(None,atom.get_var1(), atom.get_var2(),False,atom.get_name()))
			
		else:
			print("SYNTHAX ERROR")

	#Return a Query-object
	return Query(head, QueryBody(new_body), dictionary_of_variables, query_structure)

def parse_head(head_string,dictionary_of_variables):
	is_distinguished = True 
	return parse_atom(head_string,is_distinguished, dictionary_of_variables) 

def parse_body(body_string, dictionary_of_variables):
	is_distinguished = False
	atom_str_list = body_string.split("^")
	return [parse_atom(atom_str,is_distinguished, dictionary_of_variables) for atom_str in atom_str_list] 

def parse_atom(atom_string, is_distinguished, dictionary_of_variables):
	iri, arity, entry_str_list = extract_entry_tokens(atom_string)
	return AtomParser(iri, [parse_entry(token, is_distinguished, dictionary_of_variables) for token in entry_str_list])

def parse_entry(entry_string, is_distinguished, dictionary_of_variables):
	if entry_string.startswith("?"):
		return Variable(entry_string, parse_dict_of_variables(entry_string, is_distinguished, dictionary_of_variables))
	return Constant(entry_string)


#UTILITIES
def extract_entry_tokens(atom_string):
	entries_list = list()
	arity = 0
	iri = ""

	#if a atom with arity = 0 (a constant)
	if (not ("(") in atom_string):
		arity = 0
		iri = atom_string

	#if a atom with arity = 0, but with empty parantheses
	elif (("()") in atom_string):
		arity = 0
		iri = atom_string.split(("("))[0]

	else:
	#if the atom contains entries
	#else if ((("(") in atom_string) and (("?" in atom_string) or ("_" in atom_string))):
		iri, entries = atom_string.split("(", 1)
		entries = entries.replace(" ", "")
		entries = entries.rstrip(")")

		if not ((",") in entries):
			arity = 1
			entries_list.append(entries)
		else:
			entries = entries.split(",")
			for e in entries:
				entries_list.append(e)
			arity = len(entries)

	return iri,arity,entries_list

def parse_dict_of_variables(entry_string, is_distinguished, dictionary_of_variables):
	
	# If the variable is known
	if entry_string in dictionary_of_variables:
		if dictionary_of_variables[entry_string]['in_body']:
			dictionary_of_variables[entry_string]['is_shared'] = True
		else:
			dictionary_of_variables[entry_string]['in_body'] = True

	#If the variable is new
	else:
		if is_distinguished:
			dictionary_of_variables[entry_string] = {'is_bound':False, 'is_distinguished':True, 'in_body': True, 'is_shared':False }
		else:
			dictionary_of_variables[entry_string] = {'is_bound':False, 'is_distinguished':False, 'in_body': True, 'is_shared':False }
		
	if dictionary_of_variables[entry_string]['is_shared'] or dictionary_of_variables[entry_string]['is_distinguished']:
		dictionary_of_variables[entry_string]['is_bound'] = True

	return dictionary_of_variables[entry_string]

def initial_update_entries(atom, dict_of_variables):
	
	for entry in atom.get_entries():
		e = dict_of_variables[entry.get_org_name()]
		entry.update_values(e['is_distinguished'], e['in_body'], e['is_shared'])

def update_processed_status(current_q, PR, stat):
	for qu in PR:
		if current_q == qu:
			qu.set_process_status(stat)


##UPDATE
def update_body(body):
	dictionary_of_variables = {}
	for g in body.get_body():
		update_atom(g, dictionary_of_variables)

	[initial_update_entries(b, dictionary_of_variables) for b in body.get_body()]


	return body

def update_atom(atom, dictionary_of_variables):
	if isinstance(atom, AtomConcept):
		update_concept(atom.get_var1(), dictionary_of_variables)
	else:
		update_role(atom.get_var1(), atom.get_var2(), dictionary_of_variables)
	
def update_concept(var, dictionary_of_variables):
	iri = var.get_org_name()
	parse_dict_of_variables(iri, var.get_distinguished(), dictionary_of_variables)

def update_role(var1, var2, dictionary_of_variables):
	iri1 = var1.get_org_name()
	iri2 = var2.get_org_name()
	parse_dict_of_variables(iri1, var1.get_distinguished(), dictionary_of_variables)
	parse_dict_of_variables(iri2, var2.get_distinguished(), dictionary_of_variables)

def get_name_and_namespace(q, tbox):
	classes = list(tbox.classes())
	properties = list(tbox.properties())

	#for every atom in q
	for g in q.get_body().get_body():
            
		#Classes
		matches = list()
		counter = 0

        #for every class
		for cl in classes:

            #if atom name equals class name
			if g.get_iri() == cl.iri:

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
			if g.get_iri() == pp.iri:

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
	

# OTHER UTILITY PARSER METHODS
def parse_entailed_queries(entailed_queries):
    parsed_queries = list()
    for query in entailed_queries:
        new_query = dict()
        parsed_atoms = list()
        for atom in query.get_body():
            new_atom = dict()
            new_atom['iri'] = atom.get_iri()
            new_atom['name'] = atom.get_name()
            if isinstance(atom, pr.AtomConcept):
                new_atom['type'] = "concept"
            if isinstance(atom, pr.AtomRole):
                new_atom['type'] = "role"

            if new_atom['type'] == "concept":
                temp = dict()
                temp_obj = dict()
                temp['name'] = atom.get_var1().get_represented_name()
                temp['is_bound'] = atom.get_var1().get_bound()
                new_atom['var1'] = temp
                temp_obj['obj'] = new_atom
                temp_obj['str'] = ""
            if new_atom['type'] == "role":
                temp1 = dict()
                temp2 = dict()
                temp_obj = dict()
                temp1['name'] = atom.get_var1().get_represented_name()
                temp1['is_bound'] = atom.get_var1().get_bound()
                new_atom['var1'] = temp1
                temp2['name'] = atom.get_var2().get_represented_name()
                temp2['is_bound'] = atom.get_var2().get_bound()
                new_atom['var2'] = temp2
                temp_obj['obj'] = new_atom
                temp_obj['str'] = ""
            parsed_atoms.append(temp_obj)
        new_query['obj']=parsed_atoms
        parsed_queries.append(new_query)
    return parsed_queries

def parse_generated_queries(queries_from_gen):
    
    for query_structure in queries_from_gen.keys():
        for query in queries_from_gen[query_structure]:
            break


    return None

def query_reformulate(parsed_generated_queries, full_pth, t_box_path):
	#Reformulate
	# if exists
	if not os.path.exists(full_pth):      
		#for each query structure
		for query_structure in parsed_generated_queries.keys():

			#for each query in that structure
			for query_dict in parsed_generated_queries[query_structure]:

				print("Performing PerfectRef rewriting for structure " + query_structure + "...")

				#if the query structure is not up or 2u
				if not (query_structure == 'up' or query_structure == '2u'):
				
					#perform PerfectRef
					query_dict['rewritings'] = pr.get_entailed_queries(t_box_path, query_dict['q1'], False)
				else:
					temp1 = pr.get_entailed_queries(t_box_path, query_dict['q1'], False)
					temp2 = pr.get_entailed_queries(t_box_path, query_dict['q2'], False)
					query_dict['rewritings'] = temp1 + temp2

		
		with open(full_pth, 'wb') as handle:
			pickle.dump(parsed_generated_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		print("\n Reformulation already exists. Loaded pickle for this configuration. Delete or rename the pickle file if you want to redo the reformulation. \n")
		with open(full_pth, 'rb') as handle:
			parsed_generated_queries = pickle.load(handle)

	return parsed_generated_queries

# query_1p = "q(?w) :- <http://dbpedia.org/ontology/Location>(?w)"
# query_2p = "q(?w) :- <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>(?y,?x)^<http://dbpedia.org/ontology/leaderName>(?y,?w)"
# query_3p = "q(?w) :- <http://dbpedia.org/ontology/distributor>(?y,?x)^<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>(?y,?z)^<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>(?w,?z)"
# query_2i = "q(?w) :- <http://dbpedia.org/ontology/Species>(?w)^<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>(?x,?w)"
# query_3i = "q(?w) :- <http://schema.org/Place>(?w)^<http://dbpedia.org/ontology/Settlement>(?w)^<http://dbpedia.org/ontology/starring>(?w,?x)"
# query_ip = "q(?w) :- <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>(?x,?z)^<http://dbpedia.org/ontology/populationPlace>(?y,?z)^<http://www.w3.org/2000/01/rdf-schema#seeAlso>(?w, ?z)"
# query_up = "q(?z):- <https://www.wikidata.org/wiki/Q7565>(?z) | <https://www.wikidata.org/prop/P22>(?z,?y)^<https://www.wikidata.org/prop/P40>(?w, ?z)"

#tbox = owlready2.get_ontology("dataset/dbpedia15k/tbox/dbpedia15k.owl").load()
#up = query_structure_parsing(query_2p, '2p', tbox)
#print("stop")