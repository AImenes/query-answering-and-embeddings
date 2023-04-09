import owlready2
import pandas as pd
import random
import json


def get_configuration(dataset, query_structure, query_structure_length, weights, wanted_structure = None):
    df_family = pd.read_csv("dataset/family/abox/transductive/all.txt", delimiter='\t', header=None)
    df_dbpedia15k = pd.read_csv("dataset/dbpedia15k/abox/transductive/all.txt", delimiter='\t', header=None)
    
    query_length = random.randint(1,3)

    #1p
    if query_length == 1:
        currenct_structure = query_structure[1]

    # 2p, 2i, 2u
    elif query_length == 2:
        idx = random.choice(query_structure_length[2])
        currenct_structure = query_structure[idx]
    
    # 3p, 3i, pi, ip, up
    else:
        idx = random.choice(query_structure_length[3])
        currenct_structure = query_structure[idx]

    #Now, we have a randomly selected structure, currenct_structure, where the probability of each length is equal.

    if not wanted_structure is None:
        currenct_structure = wanted_structure


    #weights
    concepts, w_c = list(weights[dataset]['concepts'].keys()),list(weights[dataset]['concepts'].values())
    roles, w_r = list(weights[dataset]['roles'].keys()),list(weights[dataset]['roles'].values())

    #1p - 3 subversions
    if currenct_structure == '1p':
    
        #Both concept and roles are compatible
        is_concept = bool(random.getrandbits(1))

        if is_concept:
            random_concept = random.choices(concepts, weights=w_c)[0]

            #get answer
            return "q(?w) :- " + random_concept + "(?w)"
        else:
            random_role = random.choices(roles, weights=w_r)[0]
            is_domain = bool(random.getrandbits(1))
            if is_domain:
                return "q(?w) :- " + random_role + "(?w,?x)"
            else:
                return "q(?w) :- " + random_role + "(?x,?w)"
            
    #2p - 4 subversions
    if currenct_structure == '2p':

        #first atom has to be a role
        first_atom = random.choices(roles, weights=w_r)[0]
        is_domain = bool(random.getrandbits(1))
        if is_domain:
            #second atom
            second_atom = random.choices(roles, weights=w_r)[0]
            is_domain = bool(random.getrandbits(1))
            if is_domain:
                return "q(?w) :- " + first_atom + "(?x,?y)^" + second_atom + "(?y,?w)"
            else:
                return "q(?w) :- " + first_atom + "(?x,?y)^" + second_atom + "(?w,?y)"
        else:
            #second atom
            second_atom = random.choices(roles, weights=w_r)[0]
            is_domain = bool(random.getrandbits(1))
            if is_domain:
                return "q(?w) :- " + first_atom + "(?y,?x)^" + second_atom + "(?y,?w)"
            else:
                return "q(?w) :- " + first_atom + "(?y,?x)^" + second_atom + "(?w,?y)"
          
    #3p - 8 subversions
    if currenct_structure == '3p':

        #first atom has to be a role
        first_atom = random.choices(roles, weights=w_r)[0]
        is_domain = bool(random.getrandbits(1))
        if is_domain:
            second_atom = random.choices(roles, weights=w_r)[0]
            is_domain = bool(random.getrandbits(1))
            if is_domain:
                #third atom
                third_atom = random.choices(roles, weights=w_r)[0]
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?x,?y)^" + second_atom + "(?y,?z)^" + third_atom + "(?z,?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?x,?y)^" + second_atom + "(?y,?z)^" + third_atom + "(?w,?z)"
            else:
            #third atom
                third_atom = random.choices(roles, weights=w_r)[0]
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?x,?y)^" + second_atom + "(?z,?y)^" + third_atom + "(?z,?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?x,?y)^" + second_atom + "(?z,?y)^" + third_atom + "(?w,?z)"
        else:
            second_atom = random.choices(roles, weights=w_r)[0]
            is_domain = bool(random.getrandbits(1))
            if is_domain:
                #third atom
                third_atom = random.choices(roles, weights=w_r)[0]
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?y,?x)^" + second_atom + "(?y,?z)^" + third_atom + "(?z,?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?y,?x)^" + second_atom + "(?y,?z)^" + third_atom + "(?w,?z)"
                
            else:
                #third atom
                third_atom = random.choices(roles, weights=w_r)[0]
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?y,?x)^" + second_atom + "(?z,?y)^" + third_atom + "(?z,?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?y,?x)^" + second_atom + "(?z,?y)^" + third_atom + "(?w,?z)"
                
    #2i - 7 subversions
    if currenct_structure == '2i':
        is_concept = bool(random.getrandbits(1))

        #if first atom is a concept
        if is_concept:
            first_atom = random.choices(concepts, weights=w_c)[0]
            
            # if second atom is a concept
            is_concept = bool(random.getrandbits(1))
            if is_concept:
                second_atom = random.choices(concepts, weights=w_c)[0]
                return "q(?w) :- " + first_atom + "(?w)^" + second_atom + "(?w)"
            
            # if second atom is a role
            else:
                second_atom = random.choices(roles, weights=w_r)[0]
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?w)^" + second_atom + "(?w,?x)"
                else:
                    return "q(?w) :- " + first_atom + "(?w)^" + second_atom + "(?x,?w)"

        #if first atom is a role
        else:
            first_atom = random.choices(roles, weights=w_r)[0]
            second_atom = random.choices(roles, weights=w_r)[0]
            is_domain = bool(random.getrandbits(1))
            if is_domain:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?x,?w)^" + second_atom + "(?y,?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?x,?w)^" + second_atom + "(?w,?y)"
            else:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?w,?x)^" + second_atom + "(?y,?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?w,?x)^" + second_atom + "(?w,?y)"
            
    #3i - 15 subversions
    if currenct_structure == '3i':
        is_concept = bool(random.getrandbits(1))

        #if first atom is a concept
        if is_concept:
            first_atom = random.choices(concepts, weights=w_c)[0]
            
            # if second atom is a concept
            is_concept = bool(random.getrandbits(1))
            if is_concept:
                second_atom = random.choices(concepts, weights=w_c)[0]

                # if third atom is a concept
                is_concept = bool(random.getrandbits(1))
                if is_concept:
                    third_atom = random.choices(concepts, weights=w_c)[0]
                    return "q(?w) :- " + first_atom + "(?w)^" + second_atom + "(?w)^" + third_atom + "(?w)"
                else:
                    third_atom = random.choices(roles, weights=w_r)[0]
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?w)^" + second_atom + "(?w)^" + third_atom + "(?w,?x)"
                    else:
                        return "q(?w) :- " + first_atom + "(?w)^" + second_atom + "(?w)^" + third_atom + "(?x,?w)"
            
            # if second atom is a role
            else:
                second_atom = random.choices(roles, weights=w_r)[0]
                third_atom = random.choices(roles, weights=w_r)[0]
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?w)^" + second_atom + "(?w,?y)^" + third_atom + "(?w,?x)"
                    else:
                        return "q(?w) :- " + first_atom + "(?w)^" + second_atom + "(?w,?y)^" + third_atom + "(?x,?w)"
                else:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?w)^" + second_atom + "(?y,?w)^" + third_atom + "(?w,?x)"
                    else:
                        return "q(?w) :- " + first_atom + "(?w)^" + second_atom + "(?y,?w)^" + third_atom + "(?x,?w)"
        else:
            first_atom = random.choices(roles, weights=w_r)[0]
            second_atom = random.choices(roles, weights=w_r)[0]
            third_atom = random.choices(roles, weights=w_r)[0]
            is_domain = bool(random.getrandbits(1))
            if is_domain:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?w,?z)^" + second_atom + "(?w,?y)^" + third_atom + "(?w,?x)"
                    else:
                        return "q(?w) :- " + first_atom + "(?w,?z)^" + second_atom + "(?w,?y)^" + third_atom + "(?x,?w)"
                else:
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?w,?z)^" + second_atom + "(?y,?w)^" + third_atom + "(?w,?x)"
                    else:
                        return "q(?w) :- " + first_atom + "(?w,?z)^" + second_atom + "(?y,?w)^" + third_atom + "(?x,?w)"
            else:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?x) :- " + first_atom + "(?z,?w)^" + second_atom + "(?w,?y)^"  + third_atom + "(?w,?x)"
                    else:
                        return "q(?w) :- " + first_atom + "(?z,?w)^" + second_atom + "(?w,?y)^"  + third_atom + "(?x,?w)"
                else:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?z,?w)^" + second_atom + "(?y,?w)^" + third_atom + "(?w,?x)"
                    else:
                        return "q(?w) :- " + first_atom + "(?z,?w)^" + second_atom + "(?y,?w)^" + third_atom + "(?x,?w)"

    #pi - 6 subversions
    if currenct_structure == 'pi':

        #has to be a role
        first_atom = random.choices(roles, weights=w_r)[0]

        is_domain = random.choices(roles, weights=w_r)[0]
        if is_domain:
            second_atom = random.choices(roles, weights=w_r)[0]
            third_atom = random.choices(roles, weights=w_r)[0]
            is_domain = bool(random.getrandbits(1))
            if is_domain:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?x,?y)^" + second_atom + "(?y,?w)^" + third_atom + "(?z,?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?x,?y)^" + second_atom + "(?y,?w)^" + third_atom + "(?w,?z)"
            else:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?x,?y)^" + second_atom + "(?w,?y)^" + third_atom + "(?z,?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?x,?y)^" + second_atom + "(?w,?y)^" + third_atom + "(?w,?z)"
        else:
            second_atom = random.choices(roles, weights=w_r)[0]
            third_atom = random.choices(roles, weights=w_r)[0]
            is_domain = bool(random.getrandbits(1))
            if is_domain:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?y,?x)^" + second_atom + "(?y,?w)^" + third_atom + "(?z,?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?y,?x)^" + second_atom + "(?y,?w)^" + third_atom + "(?w,?z)"
            else:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?y,?x)^" + second_atom + "(?w,?y)^" + third_atom + "(?z,?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?y,?x)^" + second_atom + "(?w,?y)^" + third_atom + "(?w,?z)"

    #ip - 14 subversions
    if currenct_structure == 'ip':
        is_concept = bool(random.getrandbits(1))

        if is_concept:
            first_atom = random.choices(concepts, weights=w_c)[0]
            is_concept = bool(random.getrandbits(1))

            if is_concept:
                second_atom = random.choices(concepts, weights=w_c)[0]
                third_atom = random.choices(roles, weights=w_r)[0]
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?x)^" + second_atom + "(?x)^" + third_atom + "(?x, ?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?x)^" + second_atom + "(?x)^" + third_atom + "(?w, ?x)"
            else:
                second_atom = random.choices(roles, weights=w_r)[0]
                third_atom = random.choices(roles, weights=w_r)[0]
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?z)^" + second_atom + "(?y,?z)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?z)^" + second_atom + "(?y,?z)^" + third_atom + "(?w, ?z)"
                else:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?z)^" + second_atom + "(?z,?y)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?z)^" + second_atom + "(?z,?y)^" + third_atom + "(?w, ?z)"
        else:
            first_atom = random.choices(roles, weights=w_r)[0]
            second_atom = random.choices(roles, weights=w_r)[0]
            third_atom = random.choices(roles, weights=w_r)[0]
            
            is_domain = bool(random.getrandbits(1))
            if is_domain:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?x,?z)^" + second_atom + "(?y,?z)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?x,?z)^" + second_atom + "(?y,?z)^" + third_atom + "(?w, ?z)"
                else:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?x,?z)^" + second_atom + "(?z,?y)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?x,?z)^" + second_atom + "(?z,?y)^" + third_atom + "(?w, ?z)"
            else:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?z,?x)^" + second_atom + "(?y,?z)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?z,?x)^" + second_atom + "(?y,?z)^" + third_atom + "(?w, ?z)"
                else:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?z,?x)^" + second_atom + "(?z,?y)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?z,?x)^" + second_atom + "(?z,?y)^" + third_atom + "(?w, ?z)"

    #up - 14 subversions
    if currenct_structure == 'up':
        is_concept = bool(random.getrandbits(1))

        if is_concept:
            first_atom = random.choices(concepts, weights=w_c)[0]
            is_concept = bool(random.getrandbits(1))

            if is_concept:
                second_atom = random.choices(concepts, weights=w_c)[0]
                third_atom = random.choices(roles, weights=w_r)[0]
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?x) | " + second_atom + "(?x)^" + third_atom + "(?x, ?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?x) | " + second_atom + "(?x)^" + third_atom + "(?w, ?x)"
            else:
                second_atom = random.choices(roles, weights=w_r)[0]
                third_atom = random.choices(roles, weights=w_r)[0]
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?z) | " + second_atom + "(?y,?z)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?z) | " + second_atom + "(?y,?z)^" + third_atom + "(?w, ?z)"
                else:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?z) | " + second_atom + "(?z,?y)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?z) | " + second_atom + "(?z,?y)^" + third_atom + "(?w, ?z)"
        else:
            first_atom = random.choices(roles, weights=w_r)[0]
            second_atom = random.choices(roles, weights=w_r)[0]
            third_atom = random.choices(roles, weights=w_r)[0]
            
            is_domain = bool(random.getrandbits(1))
            if is_domain:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?x,?z) | " + second_atom + "(?y,?z)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?x,?z) | " + second_atom + "(?y,?z)^" + third_atom + "(?w, ?z)"
                else:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?x,?z) | " + second_atom + "(?z,?y)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?x,?z) | " + second_atom + "(?z,?y)^" + third_atom + "(?w, ?z)"
            else:
                is_domain = bool(random.getrandbits(1))
                if is_domain:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?z,?x) | " + second_atom + "(?y,?z)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?z,?x) | " + second_atom + "(?y,?z)^" + third_atom + "(?w, ?z)"
                else:
                    is_domain = bool(random.getrandbits(1))
                    if is_domain:
                        return "q(?w) :- " + first_atom + "(?z,?x) | " + second_atom + "(?z,?y)^" + third_atom + "(?z, ?w)"
                    else:
                        return "q(?w) :- " + first_atom + "(?z,?x) | " + second_atom + "(?z,?y)^" + third_atom + "(?w, ?z)"

    #2u - 7 subversions
    if currenct_structure == '2u':
        is_concept = bool(random.getrandbits(1))

        if is_concept:
            first_atom = random.choices(concepts, weights=w_c)[0]
            is_concept = bool(random.getrandbits(1))

            if is_concept:
                second_atom = random.choices(concepts, weights=w_c)[0]
                return "q(?w) :- " + first_atom + "(?w) | " + second_atom + "(?w)"
            else:
                second_atom = random.choices(roles, weights=w_r)[0] 
                is_domain = bool(random.getrandbits(1))  
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?w) | " + second_atom + "(?x, ?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?w) | " + second_atom + "(?w, ?x)"
        else:
            first_atom = random.choices(roles, weights=w_r)[0] 
            second_atom = random.choices(roles, weights=w_r)[0] 
            is_domain = bool(random.getrandbits(1))  
            if is_domain:
                is_domain = bool(random.getrandbits(1))  
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?x,?w) | " + second_atom + "(?y, ?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?x,?w) | " + second_atom + "(?w, ?y)"   
            else:
                is_domain = bool(random.getrandbits(1))  
                if is_domain:
                    return "q(?w) :- " + first_atom + "(?w,?x) | " + second_atom + "(?y, ?w)"
                else:
                    return "q(?w) :- " + first_atom + "(?w,?x) | " + second_atom + "(?w, ?y)"   
                 
def get_weights():

    # Get the weights from the ABox
    df_family = pd.read_csv("dataset/family/abox/transductive/all.txt", delimiter='\t', header=None)
    df_dbpedia15k = pd.read_csv("dataset/dbpedia15k/abox/transductive/all.txt", delimiter='\t', header=None)

    #Using pandas way, Series.value_counts()
    df_family_roles = df_family.iloc[:,1].value_counts()
    df_dbpedia15k_roles = df_dbpedia15k.iloc[:,1].value_counts()

    #Count concepts in dbpedia15k. That is, do a value count on the objects in the triples where rdf:type is the property.
    df_dbpedia15k_rdftype = df_dbpedia15k.loc[df_dbpedia15k[1] == df_dbpedia15k_roles.index.tolist()[0]]
    df_dbpedia15k_concepts = df_dbpedia15k_rdftype.iloc[:,2].value_counts()

    tbox_family     = owlready2.get_ontology("dataset/family/tbox/family_wikidata.owl").load()    
    tbox_dbpedia15k = owlready2.get_ontology("dataset/dbpedia15k/tbox/dbpedia15k.owl").load()
    classes_family = list(tbox_family.classes())
    properties_family = list(tbox_family.object_properties())
    classes_dbpedia15k = list(tbox_dbpedia15k.classes())
    properties_dbpedia15k = list(tbox_dbpedia15k.object_properties())
    
    dbpedia15k_concepts = df_dbpedia15k_concepts.to_dict()
    dbpedia15k_roles = df_dbpedia15k_roles.to_dict()
    family_classes = dict()
    family_roles = df_family_roles.to_dict()

    for cl in classes_dbpedia15k:
        iri = "<" + cl.iri + ">"
        if not iri in dbpedia15k_concepts:
            dbpedia15k_concepts[iri] = 1

    for pr in properties_dbpedia15k:
        iri = "<" + pr.iri + ">"
        if not iri in dbpedia15k_roles:
            dbpedia15k_roles[iri] = 1

    for cl in classes_family:
        iri = "<" + cl.iri + ">"
        if not iri in family_classes:
            family_classes[iri] = 1

    for pr in properties_family:
        iri = "<" + pr.iri + ">"
        if not iri in family_roles:
            family_roles[iri] = 1
        
    #Removing RDF-type from query-structures
    del dbpedia15k_roles['<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>']
    del dbpedia15k_roles['<http://www.w3.org/2000/01/rdf-schema#seeAlso>']

    return {'dbpedia15k': 
                {'concepts': dbpedia15k_concepts, 'roles': dbpedia15k_roles},
            'family':
                {'concepts': family_classes, 'roles': family_roles}
            }

def query_gen(testcase, number_of_queries_per_structure):
    #init
    query_structure = {1: '1p', 2: '2p', 3: '3p', 4: '2i', 5: '3i', 6: 'pi', 7: 'ip', 8: 'up', 9: '2u'}
    query_length = {1: [1], 2: [2, 4, 9], 3: [3,5,6,7,8]}
    queries = dict()
    weights = get_weights()
    datasets = ['family', 'dbpedia15k']

    #generate queries
    for ds in datasets:
        for structure in query_structure.values():
            i = 0
            temp_list = list()
            while (i < number_of_queries_per_structure):
                temp_list.append(get_configuration(ds, query_structure, query_length, weights, structure))
                i += 1
            queries[structure] = temp_list

        file_name = "testcases/" + testcase + "/queries/" + ds + '/queries_' + "k-" + str(number_of_queries_per_structure) + '.txt'
        with open(file_name, 'w') as convert_file:
            convert_file.write(json.dumps(queries, indent=2))




