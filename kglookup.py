import numpy as np
import pandas as pd

# Import PerfectRef and OwlReady2
import perfectref_v1 as pr
from perfectref_v1 import Query, QueryBody
from perfectref_v1 import AtomParser, AtomConcept, AtomRole, AtomConstant
from perfectref_v1 import Variable, Constant
from owlready2 import get_ontology

def kg_lookup(queries, ds, abox_path, tf):
    abox = abox_path + "all.txt"
    columns = ['head', 'relation', 'tail']
    abox = pd.read_csv(abox, sep='\t', names=columns, header=None)
    
    for structure, query_list in queries.items():
        print("Looking up answers for %s-queries." % ((structure)))
        if structure == '1p' or structure == '2p' or structure == '3p' or structure == 'pi':

            for query in query_list:
                distinguished_var = query['q1'].head.entries[0].original_entry_name
                query_length = len(query['q1'].get_body().get_body())
                i = 0
                projection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                entities_to_merge = list()
                for g in query['q1'].get_body().get_body():
                    atom = {'type': None, 'variable': None, 'target': None, 'entities': None}
                    #Last atom
                    if i == query_length - 1:
                        last_atom = True
                    else:
                        last_atom = False

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
                    query['kglookup'] = entities_to_merge[0]
                elif len(entities_to_merge) == 2:
                    query['kglookup'] = list(set(entities_to_merge[0]) & set(entities_to_merge[1]))
                elif len(entities_to_merge) == 3:
                    query['kglookup'] = set(entities_to_merge[0]) & set(entities_to_merge[1]) & set(entities_to_merge[2])
                else:
                    print("An error has occured")

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

        if structure == 'ip':
             for query in query_list:
                distinguished_var = query['q1'].head.entries[0].original_entry_name
                query_length = len(query['q1'].get_body().get_body())
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
                            #intersection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                        elif g.var2.get_org_name() == distinguished_var:
                            atom['target'] = 'tail'
                            entities_to_merge.append(atom['entities'][distinguished_var])
                            #intersection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                        
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

        if structure == 'up':
            for query in query_list:
                distinguished_var = query['q1'].head.entries[0].original_entry_name
                query_length = len(query['q1'].get_body().get_body())
                i = 0
                projection = {'type': None, 'variable': None, 'target': None, 'entities': None}
                entities_to_merge = list()
                for g in query['q1'].get_body().get_body():
                    atom = {'type': None, 'variable': None, 'target': None, 'entities': None}
                    #Last atom
                    if i == query_length - 1:
                        last_atom = True
                    else:
                        last_atom = False

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
                    #Last atom
                    if i == query_length - 1:
                        last_atom = True
                    else:
                        last_atom = False

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

def query_graph_concepts(query, dataset, a_box_path):
    entities_to_return = list()
    #test = pr.parse_query(query["str"]).get_body().get_body()

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