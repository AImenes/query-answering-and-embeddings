import numpy as np
import pandas as pd
import requests
import time
import json

# Import PerfectRef and OwlReady2
from perfectref_v1 import AtomParser, AtomConcept, AtomRole, AtomConstant, QueryBody


def kg_lookup(queries, ds, abox_path, tf):
    """
    This method extracts the entities from the local KG which the KGEs models are trained upon. They are a little outdated from
    the online versions.

    Args:
        queries (dict):         The dictionary containing all queries for all structures.
        ds (str):               The dataset name
        abox_path (str):        The relative path to the entire abox of the dataset, which we will to KG-lookup.
        tf (TriplesFactory):    The TriplesFactory used together with it.

    Returns:
        queries (dict):         Returns an updated queries dictionary also containing the kglookup results.
    """
    
    # Load ABox
    abox = abox_path + "all.txt"
    columns = ['head', 'relation', 'tail']
    abox = pd.read_csv(abox, sep='\t', names=columns, header=None)
    
    # Iterate through the structure types and the queries inside them.
    for structure, query_list in queries.items():
        print("Looking up answers for %s-queries." % ((structure)))
        
        # If the structure is a projection
        if structure == '1p' or structure == '2p' or structure == '3p' or structure == 'pi':
            
            # Iterate through every query in a structure
            for query in query_list:
                distinguished_var = query['q1'].head.entries[0].original_entry_name
                i = 0
                projection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                
                # The list we will merge in the end of the method
                entities_to_merge = list()
                
                # Iterate through every atom in a query
                for g in query['q1'].body.body:
                    atom = {'type': None, 'variable': None, 'target': None, 'entities': None}

                    # Query using the concept method if the atom is a Concept.
                    if isinstance(g, AtomConcept):
                        atom['type'] = 'concept'
                        try:
                            atom['entities'] = [t[0] for t in query_graph_concepts(g, ds, abox_path)]
                        except:
                            atom['entities'] = []

                        if g.var1.original_entry_name == distinguished_var:
                            atom['variable'] = g.var1.original_entry_name
                            atom['target'] = 'head'
                            
                            entities_to_merge.append(atom['entities'])
                    
                    if isinstance(g, AtomRole):
                        atom['type'] = 'role'
                        atom['entities'] = query_graph_roles(g, tf, abox, True, projection)
                        
                        if g.var1.original_entry_name == distinguished_var:
                            atom['target'] = 'head'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                            projection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                        elif g.var2.original_entry_name == distinguished_var:
                            atom['target'] = 'tail'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                            projection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                        
                        #Projection
                        else:
                            #if the variable is shared but not the previous shared
                            if g.var1.shared and g.var1.original_entry_name != projection['variable']:
                                projection['type'] = 'role'
                                projection['variable'] = g.var1.original_entry_name
                                projection['target'] = 'head'
                                projection['entities'] = atom['entities'][g.var1.original_entry_name]
                            elif g.var2.shared and g.var2.original_entry_name != projection['variable']:
                                projection['type'] = 'role'
                                projection['variable'] = g.var2.original_entry_name
                                projection['target'] = 'tail'
                                projection['entities'] = atom['entities'][g.var2.original_entry_name]
                                
                             
                    i += 1

                #Merge, if necessary
                if len(entities_to_merge) == 1:
                    query['kglookup'] = entities_to_merge[0]
                elif len(entities_to_merge) == 2:
                    query['kglookup'] = list(set(entities_to_merge[0]) & set(entities_to_merge[1]))
                elif len(entities_to_merge) == 3:
                    query['kglookup'] = set(entities_to_merge[0]) & set(entities_to_merge[1]) & set(entities_to_merge[2])
                else:
                    print("An error has occured")

        # For structures that are intersective
        if structure == '2i' or structure == '3i':
            for query in query_list:
                distinguished_var = query['q1'].head.entries[0].original_entry_name
                query_length = len(query['q1'].get_body().get_body())
                i = 0
                entities_to_merge = list()
                for g in query['q1'].get_body().get_body():
                    atom = {'type': None, 'variable': None, 'target': None, 'entities': None}

                    if isinstance(g, AtomConcept):
                        atom['type'] = 'concept'
                        try:
                            atom['entities'] = [t[0] for t in query_graph_concepts(g, ds, abox_path)]
                        except:
                            atom['entities'] = [] 
                        if g.var1.get_org_name() == distinguished_var:
                            atom['variable'] = g.var1.get_org_name()
                            atom['target'] = 'head'
                                                             
                            entities_to_merge.append(atom['entities'])
                    if isinstance(g, AtomRole):
                        atom['type'] = 'role'
                        atom['entities'] = query_graph_roles(g, tf, abox, False)
                        
                        if g.var1.get_org_name() == distinguished_var:
                            atom['target'] = 'head'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                        elif g.var2.get_org_name() == distinguished_var:
                            atom['target'] = 'tail'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                             
                    i += 1

                #Merge, if necessary
                if len(entities_to_merge) == 1:
                    query['kglookup'] = entities_to_merge[0]
                elif len(entities_to_merge) == 2:
                    query['kglookup'] = list(set(entities_to_merge[0]) & set(entities_to_merge[1]))
                elif len(entities_to_merge) == 3:
                    query['kglookup'] = list(set(entities_to_merge[0]) & set(entities_to_merge[1]) & set(entities_to_merge[2]))
                else:
                    print("An error has occured")

        # If the structure is both intersective and projective
        if structure == 'ip':
             for query in query_list:
                distinguished_var = query['q1'].head.entries[0].original_entry_name
                i = 0
                intersection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                entities_to_merge = list()
                for g in query['q1'].get_body().get_body():
                    atom = {'type': None, 'variable': None, 'target': None, 'entities': None}

                    if isinstance(g, AtomConcept):
                        atom['type'] = 'concept'
                        try:
                            atom['entities'] = [t[0] for t in query_graph_concepts(g, ds, abox_path)]
                        except:
                            atom['entities'] = []

                        if g.var1.get_org_name() == distinguished_var:
                            atom['variable'] = g.var1.get_org_name()
                            atom['target'] = 'head'
                            entities_to_merge.append(atom['entities'])

                        else:
                            if g.var1.shared and g.var1.original_entry_name == intersection['variable']:
                                intersection['type'] = 'concept'
                                intersection['variable'] = g.var1.original_entry_name
                                intersection['target'] = 'head'
                                intersection['entities'] = list(set(intersection['entities']) & set(atom['entities']))

                            #First atom
                            else:
                                if g.var1.shared and intersection['variable'] is None:
                                    intersection['type'] = 'concept'
                                    intersection['variable'] = g.var1.original_entry_name
                                    intersection['target'] = 'head'
                                    intersection['entities'] = atom['entities']
                    
                    if isinstance(g, AtomRole):
                        atom['type'] = 'role'
                        atom['entities'] = query_graph_roles(g, tf, abox, True, intersection)
                        
                        if g.var1.get_org_name() == distinguished_var:
                            atom['target'] = 'head'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                        elif g.var2.get_org_name() == distinguished_var:
                            atom['target'] = 'tail'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                        
                        #Intersection
                        else:
                            #if the variable is shared and the previous shared
                            if g.var1.shared and g.var1.original_entry_name == intersection['variable']:
                                intersection['type'] = 'role'
                                intersection['variable'] = g.var1.original_entry_name
                                intersection['target'] = 'head'
                                intersection['entities'] = list(set(intersection['entities']) & set(atom['entities'][g.var1.original_entry_name]))
                            elif g.var2.shared and g.var2.original_entry_name == intersection['variable']:
                                intersection['type'] = 'role'
                                intersection['variable'] = g.var2.original_entry_name
                                intersection['target'] = 'tail'
                                intersection['entities'] = list(set(intersection['entities']) & set(atom['entities'][g.var2.original_entry_name]))
                            
                            #First atom
                            else:
                                if g.var1.shared and intersection['variable'] is None:
                                    intersection['type'] = 'role'
                                    intersection['variable'] = g.var1.original_entry_name
                                    intersection['target'] = 'head'
                                    intersection['entities'] = atom['entities'][g.var1.original_entry_name]
                                elif g.var2.shared and intersection['variable'] is None:
                                    intersection['type'] = 'role'
                                    intersection['variable'] = g.var2.original_entry_name
                                    intersection['target'] = 'tail'
                                    intersection['entities'] = atom['entities'][g.var2.original_entry_name]
                                
                             
                    i += 1

                #Merge, if necessary
                if len(entities_to_merge) == 1:
                    query['kglookup'] = entities_to_merge[0]
                elif len(entities_to_merge) == 2:
                    query['kglookup'] = list(set(entities_to_merge[0]) & set(entities_to_merge[1]))
                elif len(entities_to_merge) == 3:
                    query['kglookup'] = set(entities_to_merge[0]) & set(entities_to_merge[1]) & set(entities_to_merge[2])
                else:
                    print("An error has occured")

        # Disjunction 2u
        if structure == '2u':
            for query in query_list:
                distinguished_var = query['q1'].head.entries[0].original_entry_name
                i = 0
                entities_to_merge = list()
                for g in query['q1'].get_body().get_body():
                    atom = {'type': None, 'variable': None, 'target': None, 'entities': None}

                    if isinstance(g, AtomConcept):
                        atom['type'] = 'concept'
                        try:
                            atom['entities'] = [t[0] for t in query_graph_concepts(g, ds, abox_path)]
                        except:
                            atom['entities'] = [] 
                        if g.var1.get_org_name() == distinguished_var:
                            atom['variable'] = g.var1.get_org_name()
                            atom['target'] = 'head'
                                                             
                            entities_to_merge.append(atom['entities'])
                    if isinstance(g, AtomRole):
                        atom['type'] = 'role'
                        atom['entities'] = query_graph_roles(g, tf, abox, False)
                        
                        if g.var1.get_org_name() == distinguished_var:
                            atom['target'] = 'head'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                        elif g.var2.get_org_name() == distinguished_var:
                            atom['target'] = 'tail'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                             
                    i += 1
                i = 0
                for g in query['q2'].get_body().get_body():
                    atom = {'type': None, 'variable': None, 'target': None, 'entities': None}

                    if isinstance(g, AtomConcept):
                        atom['type'] = 'concept'
                        try:
                            atom['entities'] = [t[0] for t in query_graph_concepts(g, ds, abox_path)]
                        except:
                            atom['entities'] = []
                        if g.var1.get_org_name() == distinguished_var:
                            atom['variable'] = g.var1.get_org_name()
                            atom['target'] = 'head'
                                                             
                            entities_to_merge.append(atom['entities'])
                    if isinstance(g, AtomRole):
                        atom['type'] = 'role'
                        atom['entities'] = query_graph_roles(g, tf, abox, False)
                        
                        if g.var1.get_org_name() == distinguished_var:
                            atom['target'] = 'head'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                        elif g.var2.get_org_name() == distinguished_var:
                            atom['target'] = 'tail'
                            entities_to_merge.append(atom['entities'][distinguished_var])

                #Merge, if necessary
                if len(entities_to_merge) == 1:
                    query['kglookup'] = entities_to_merge[0]
                elif len(entities_to_merge) == 2:
                    query['kglookup'] = list(set(entities_to_merge[0]) | set(entities_to_merge[1]))
                elif len(entities_to_merge) == 3:
                    query['kglookup'] = list(set(entities_to_merge[0]) | set(entities_to_merge[1]) | set(entities_to_merge[2]))
                else:
                    print("An error has occured")

        
        # Disjunction up
        if structure == 'up':
            for query in query_list:
                distinguished_var = query['q1'].head.entries[0].original_entry_name
                i = 0
                projection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                entities_to_merge = list()
                for g in query['q1'].get_body().get_body():
                    atom = {'type': None, 'variable': None, 'target': None, 'entities': None}

                    if isinstance(g, AtomConcept):
                        atom['type'] = 'concept'
                        try:
                            atom['entities'] = [t[0] for t in query_graph_concepts(g, ds, abox_path)]
                        except:
                            atom['entities'] = []

                        if g.var1.get_org_name() == distinguished_var:
                            atom['variable'] = g.var1.get_org_name()
                            atom['target'] = 'head'
                            

                            entities_to_merge.append(atom['entities'])
                    if isinstance(g, AtomRole):
                        atom['type'] = 'role'
                        atom['entities'] = query_graph_roles(g, tf, abox, True, projection)
                        
                        if g.var1.get_org_name() == distinguished_var:
                            atom['target'] = 'head'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                            projection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                        elif g.var2.get_org_name() == distinguished_var:
                            atom['target'] = 'tail'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                            projection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                        
                        #Projection
                        else:
                            #if the variable is shared but not the previous shared
                            if g.var1.shared and g.var1.original_entry_name != projection['variable']:
                                projection['type'] = 'role'
                                projection['variable'] = g.var1.original_entry_name
                                projection['target'] = 'head'
                                projection['entities'] = atom['entities'][g.var1.original_entry_name]
                            elif g.var2.shared and g.var2.original_entry_name != projection['variable']:
                                projection['type'] = 'role'
                                projection['variable'] = g.var2.original_entry_name
                                projection['target'] = 'tail'
                                projection['entities'] = atom['entities'][g.var2.original_entry_name]
                                
                             
                    i += 1

                    #Merge, if necessary
                if len(entities_to_merge) == 1:
                    query_1 = entities_to_merge[0]
                elif len(entities_to_merge) == 2:
                    query_1 = list(set(entities_to_merge[0]) & set(entities_to_merge[1]))
                elif len(entities_to_merge) == 3:
                    query_1 = set(entities_to_merge[0]) & set(entities_to_merge[1]) & set(entities_to_merge[2])
                else:
                    print("An error has occured")
                
                i = 0
                projection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                entities_to_merge = list()
                for g in query['q2'].get_body().get_body():
                    atom = {'type': None, 'variable': None, 'target': None, 'entities': None}

                    if isinstance(g, AtomConcept):
                        atom['type'] = 'concept'
                        try:
                            atom['entities'] = [t[0] for t in query_graph_concepts(g, ds, abox_path)]
                        except:
                            atom['entities'] = []

                        if g.var1.get_org_name() == distinguished_var:
                            atom['variable'] = g.var1.get_org_name()
                            atom['target'] = 'head'
                            

                            entities_to_merge.append(atom['entities'])
                    if isinstance(g, AtomRole):
                        atom['type'] = 'role'
                        atom['entities'] = query_graph_roles(g, tf, abox, True, projection)
                        
                        if g.var1.get_org_name() == distinguished_var:
                            atom['target'] = 'head'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                            projection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                        elif g.var2.get_org_name() == distinguished_var:
                            atom['target'] = 'tail'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                            projection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                        
                        #Projection
                        else:
                            #if the variable is shared but not the previous shared
                            if g.var1.shared and g.var1.original_entry_name != projection['variable']:
                                projection['type'] = 'role'
                                projection['variable'] = g.var1.original_entry_name
                                projection['target'] = 'head'
                                projection['entities'] = atom['entities'][g.var1.original_entry_name]
                            elif g.var2.shared and g.var2.original_entry_name != projection['variable']:
                                projection['type'] = 'role'
                                projection['variable'] = g.var2.original_entry_name
                                projection['target'] = 'tail'
                                projection['entities'] = atom['entities'][g.var2.original_entry_name]
                                
                             
                    i += 1

                #Merge, if necessary
                if len(entities_to_merge) == 1:
                    query_2 = entities_to_merge[0]
                elif len(entities_to_merge) == 2:
                    query_2 = list(set(entities_to_merge[0]) & set(entities_to_merge[1]))
                elif len(entities_to_merge) == 3:
                    query_2 = set(entities_to_merge[0]) & set(entities_to_merge[1]) & set(entities_to_merge[2])
                else:
                    print("An error has occured")
                
                #Merge each subquery
                query['kglookup'] = list(set(query_1) | set(query_2))

    return queries

def is_prediction_kg_hit(final_df, kglookup):
    """
    Args:
        final_df (pd.DataFrame): The final results dataframe
        kglookup (list): a list of all the local KG hits

    Returns:
        pd.DataFrame: a boolean column if entity is in kglookup.
    """
    return final_df['entity_label'].isin(kglookup)

def online_kg_lookup(final_df, query, dataset):
    """
    
    Given the query and the dataset, query the respective online
    KG for entity answers. Due to HTML and API limitations, some
    sleep parameters and entity splits are created to do make sure
    we don't get stuck with error responses.

    Args:
        final_df (pd.DataFrame):    The final results dataframe
        query (QueryBody or dict):  The query body of the current query. 
                                    It is a QueryBody for each rewriting, but a dict
                                    for the original query.
        dataset (str):              The name of the dataset

    Returns: 
        pd.DataFrame: a boolean column if entity is in kglookup.

    """
    entities = final_df['entity_label'].to_list()

    # Divide entities into chunks, so we dont get too long queries.
    def divide_chunks(l, n): 
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]
    chunks = 75
    entities = list(divide_chunks(entities, chunks))


    # Construct the SELECT clause with the distinguished variable
    distinguished_variable = query['q1'].head.var1.original_entry_name if isinstance(query, dict) else next(iter(query.variable_hierarchy))
    select_clause = f"SELECT {distinguished_variable} WHERE {{ "

    # For the Family dataset (WikiData)
    if dataset == 'family':
        for entity_chunk in entities:

            # Construct the VALUES clause with entities
            values_clause = "VALUES ?w { "
            for entity in entity_chunk:
                entity = entity.replace("https", "http").replace("/wiki/", "/entity/").replace("<http://www.wikidata.org/entity/", "wd:").replace(">", "")
                values_clause += f"{entity} "
            values_clause += "} "

            # If a rewritten query, construct the body of the query for q1
            if isinstance(query, QueryBody):
                query_body_q1 = ""
                for atom in query.body:
                    if isinstance(atom, AtomRole):
                        query_body_q1 += f"{atom.var1.original_entry_name} wdt:{atom.name} {atom.var2.original_entry_name} . "
                    elif isinstance(atom, AtomConcept):
                        query_body_q1 += f"{atom.var1.original_entry_name} wdt:P31 wd:{atom.name} . "

                query_body = query_body_q1
            
            # If the original query, construct the body of the query for q1
            if isinstance(query, dict):
                query_body_q1 = ""
                for atom in query['q1'].body.body:
                    if isinstance(atom, AtomRole):
                        query_body_q1 += f"{atom.var1.original_entry_name} wdt:{atom.name} {atom.var2.original_entry_name} . "
                    elif isinstance(atom, AtomConcept):
                        query_body_q1 += f"{atom.var1.original_entry_name} wdt:P31 wd:{atom.name} . "

                # If there is a second query q2, construct its body
                if query['q2'] is not None:
                    query_body_q2 = ""
                    for atom in query['q2'].body.body:
                        if isinstance(atom, AtomRole):
                            query_body_q2 += f"{atom.var1.original_entry_name} wdt:{atom.name} {atom.var2.original_entry_name} . "
                        elif isinstance(atom, AtomConcept):
                            query_body_q2 += f"{atom.var1.original_entry_name} wdt:P31 wd:{atom.name} . "
                    # Merge q1 and q2 with UNION
                    query_body = f"{{ {query_body_q1} }} UNION {{ {query_body_q2} }}"
                else:
                    query_body = query_body_q1
            
            # Close the query
            query_body += "} "

            # Construct the complete SPARQL query
            sparql_query = f"{select_clause} {values_clause} {query_body}"
            print(sparql_query)
            # Send the SPARQL query and get the response
            url = 'https://query.wikidata.org/sparql'
            r = requests.get(url, params={'format': 'json', 'query': sparql_query})

            while not r.status_code == 200:
                print("Sleep 1 minute for Online lookup restoring.")
                time.sleep(60)
                r = requests.get(url, params={'format': 'json', 'query': sparql_query})
            
            data = r.json()

            #Construct 
            true_entities = list()
            for binding in data['results']['bindings']:
                entity = "<" + binding[distinguished_variable[1:]]['value'] + ">"
                entity = entity.replace("http:", "https:").replace("/entity/", "/wiki/")
                if not entity in true_entities:
                    true_entities.append(entity)
            
            time.sleep(1)
    
    #DBPEDIA  
    else:
        for entity_chunk in entities:

            # Construct the VALUES clause with entities
            values_clause = "VALUES ?w { "
            forbidden_characters = {'\"', 'Ä', 'ā', 'Ö', 'ī', 'æ', 'ø', 'å', 'Š', 'Ž', '\`', '`'}
            for entity in entity_chunk:
                if all(char not in forbidden_characters for char in entity):
                    values_clause += f"{entity} "
            values_clause += "} "

            # If a rewritten query, construct the body of the query for q1
            if isinstance(query, QueryBody):
                query_body_q1 = ""
                for atom in query.body:
                    if isinstance(atom, AtomRole):
                        query_body_q1 += f"{atom.var1.original_entry_name} <{atom.iri}> {atom.var2.original_entry_name} . "
                    elif isinstance(atom, AtomConcept):
                        query_body_q1 += f"{atom.var1.original_entry_name} rdf:type <{atom.iri}> . "

                query_body = query_body_q1
            
            # If the original query, construct the body of the query for q1
            if isinstance(query, dict):
                query_body_q1 = ""
                for atom in query['q1'].body.body:
                    if isinstance(atom, AtomRole):
                        query_body_q1 += f"{atom.var1.original_entry_name} <{atom.iri}> {atom.var2.original_entry_name} . "
                    elif isinstance(atom, AtomConcept):
                        query_body_q1 += f"{atom.var1.original_entry_name} rdf:type <{atom.iri}> . "

                # If there is a second query q2, construct its body
                if query['q2'] is not None:
                    query_body_q2 = ""
                    for atom in query['q2'].body.body:
                        if isinstance(atom, AtomRole):
                            query_body_q2 += f"{atom.var1.original_entry_name} <{atom.iri}> {atom.var2.original_entry_name} . "
                        elif isinstance(atom, AtomConcept):
                            query_body_q2 += f"{atom.var1.original_entry_name} rdf:type <{atom.iri}> . "
                    
                    # Merge q1 and q2 with UNION
                    query_body = f"{{ {query_body_q1} }} UNION {{ {query_body_q2} }}"
                else:
                    query_body = query_body_q1
            
            # Close the query
            query_body += "} "

            # Construct the complete SPARQL query
            sparql_query = f"{select_clause} {values_clause} {query_body}"
            print(sparql_query)
            # Send the SPARQL query and get the response
            url = 'https://dbpedia.org/sparql'
            r = requests.get(url, params={'format': 'json', 'query': sparql_query})
            
            while r.status_code == 429:
                print("Sleeping 1 minute for deload API.")
                time.sleep(60)
            
            if r.status_code == 200 or r.status_code == 206:
                data = r.json()
            else:
                raise ImportError("Error code %i from HTML response" % (r.status_code))

            #Construct 
            true_entities = list()
            for binding in data['results']['bindings']:
                entity = "<" + binding[distinguished_variable[1:]]['value'] + ">"
                if not entity in true_entities:
                    true_entities.append(entity)
            
            time.sleep(0.5)


    # Return the results
    return final_df['entity_label'].isin(true_entities)
    

def query_graph_concepts(query, dataset, a_box_path):
    entities_to_return = list()

    #if dataset has classes in separate files
    if dataset == "dbpedia15k":
        with open(a_box_path + "class2id_mod.txt") as fd:
            classes = fd.read().splitlines()

        # Check id for this class/concept
        temp = [i for i, x in enumerate(classes) if x == query.iri]

        if len(temp) == 1:
            idx = temp[0]
        else:
            return None

        # Append entities in which matches the class/concept
        entities2classes = np.genfromtxt(
            a_box_path + "entity2class.txt", dtype=int, delimiter='\t')
        matches = np.where(entities2classes[:, -1] == idx)
        tests = entities2classes[matches][:, 0]

        entity2id = pd.read_csv(
            a_box_path + "entity2id.txt", sep="\t", header=None)
        testlist = entity2id.values.tolist()

        for cl in tests:
            entities_to_return.append(testlist[cl])
        return entities_to_return   
    else:
        return None

def query_graph_roles(g, tf, abox, is_projection, candidates = {'type': None}):
    relation = tf.relations_to_ids(relations=["<"+g.iri+">"])
    triples = tf.new_with_restriction(relations=relation)
    
    if is_projection and candidates['type'] is not None:
        temp = abox.loc[abox['relation'] == "<"+g.iri+">"]
        
        #Projection
        # r1(x,y), r2(y,z)
        if (candidates['variable'] == g.var1.original_entry_name) and g.var2.bound:
            temp = temp.loc[temp['head'].isin(candidates['entities'])]
            return {g.var2.get_org_name(): list(temp['tail'].unique())}

        # r1(x,y), r2(z,y)
        elif (candidates['variable'] == g.var2.original_entry_name) and g.var1.bound:
            temp = temp.loc[temp['tail'].isin(candidates['entities'])]
            return {g.var1.get_org_name(): list(temp['head'].unique())}
        
    
    heads = np.unique(triples.mapped_triples[:,0])
    tails = np.unique(triples.mapped_triples[:,2])
    head_labels = [triples.entity_labeling.id_to_label[x] for x in heads]
    tail_labels = [triples.entity_labeling.id_to_label[x] for x in tails]

    #domain
    if g.var1.bound and g.var2.bound:
        return {g.var1.get_org_name(): head_labels, g.var2.get_org_name(): tail_labels}
    elif g.var1.bound:
        return {g.var1.get_org_name(): head_labels, g.var2.get_org_name(): None}
    elif g.var2.bound:
        return {g.var1.get_org_name(): None, g.var2.get_org_name(): tail_labels}
    else:
        return {g.var1.get_org_name(): None, g.var2.get_org_name(): None}
