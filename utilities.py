import os
import pickle
from torch.cuda import is_available
from owlready2 import get_ontology

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def press_any_key():
    holder = input("\n\nPress any key to continue ...")

def get_device():
    if is_available():
        device = 'gpu'
    else:
        device = 'cpu'
    return device

def load_ontology(path):
    ontology = get_ontology(path).load()
    return ontology

def show_tbox(tbox):
    print("CLASSES:\n", list(tbox.classes()))
    print("\n\nPROPERTIES:\n1", list(tbox.properties()))

def enter_query(ontology):

    distinguished = list()
    keep_adding_distinguished = True
    if ontology == None:
        print("Please import a tbox before using the query wizard.\n")
        press_any_key()
        return None
    while keep_adding_distinguished:
        temp = input("Enter distinguished variable name(default = x): ")
        if temp == "":
            temp = "x"
            if temp not in distinguished:
                distinguished.append(temp)
        else:
            if temp not in distinguished:
                distinguished.append(temp)
        cont = input(
            "\nIf you would like another distinguished variable, enter 1. Leave empty to finish [empty]: ")
        if cont == "":
            keep_adding_distinguished = False

    classes = list(ontology.classes())
    properties = list(ontology.properties())
    atoms = list()
    running = True

    while running:
        atom = dict()
        atom["iri"] = None
        atom['type'] = None
        counter = 0
        matches = list()

        while atom["iri"] == None:
            atom['name'] = input("\nEnter atom name:")
            # Classes
            for cl in classes:
                if atom['name'] == cl.name:
                    counter += 1
                    matches.append(cl)

            if counter == 1:
                atom['iri'] = matches[0].iri
                atom['type'] = "concept"

            elif counter > 1:
                for i in range(len(matches)):
                    print(str(i) + ":\t" + matches[i].iri)

                option = None
                while not (isinstance(option, int) and option < len(matches)):
                    option = int(
                        input("\nSelect the ID of the wanted IRI (0-" + str(len(matches)-1) + "): "))

                atom['iri'] = matches[option].iri
                atom['type'] = "concept"

            else:
                pass

            if atom['type'] == "concept":
                if len(distinguished) > 0:
                    for j in range(len(distinguished)):
                        print("[%i]:\t%s" % (j, distinguished[j]))
                    print("[%i]:\t%s" % (len(
                        distinguished), "Enter other variable, which will occur more than once in the body."))
                    print("[%i]:\t%s" % (len(distinguished)+1, "_ (Unbound)"))
                    option = int(
                        input("\nSelect the corresponding variable number: "))
                    idx = "var1"
                    temp = dict()
                    if option > len(distinguished):
                        temp['name'] = "_"
                        temp['is_bound'] = False
                    elif option == len(distinguished):
                        temp['name'] = input("\nEnter variable name: ")
                        if temp['name'] in distinguished:
                            temp['is_bound'] = True
                        else:
                            temp['is_bound'] = False
                    else:
                        temp['name'] = distinguished[option]
                        temp['is_bound'] = True
                    atom[idx] = temp
                else:
                    atom['var1'] = None

                completed_atom = dict()
                completed_atom["str"] = atom["name"] + \
                    "(?" + atom['var1']['name'] + ")"
                completed_atom["obj"] = atom
                atoms.append(completed_atom)

            if counter == 0:
                # Properties
                for pp in properties:
                    if atom['name'] == pp.name:
                        counter += 1
                        matches.append(pp)

                if counter == 1:
                    atom['iri'] = matches[0].iri
                    atom['type'] = "role"

                elif counter > 1:
                    for i in range(len(matches)):
                        print(str(i) + ":\t" + matches[i].iri)

                    option = None
                    while not (isinstance(option, int) and option < len(matches)):
                        option = int(
                            input("\nSelect the ID of the wanted IRI (0-" + str(len(matches)-1) + "): "))

                    atom['iri'] = matches[option].iri
                    atom['type'] = "role"
                else:
                    print(
                        "\nYou have entered a non existent class or property in the current TBox. Remember to be case sensitive.")
                if atom['type'] == "role":
                    for i in range(2):
                        print("What should be in the %i. variable spot?" % (i+1))

                        for j in range(len(distinguished)):
                            print("[%i]:\t%s" % (j, distinguished[j]))
                        print("[%i]:\t%s" % (len(
                            distinguished), "Enter other variable, which will occur more than once in the body."))
                        print("[%i]:\t%s" %
                              (len(distinguished)+1, "_ (Unbound)"))
                        option = int(
                            input("\nSelect the corresponding variable number: "))
                        idx = "var" + str(i+1)
                        temp = dict()
                        if option > len(distinguished):
                            temp['name'] = "_"
                            temp['is_bound'] = False
                        elif option == len(distinguished):
                            temp['name'] = input("\nEnter variable name: ")
                            if temp['name'] in distinguished:
                                temp['is_bound'] = True
                            else:
                                temp['is_bound'] = False
                        else:
                            temp['name'] = distinguished[option]
                            temp['is_bound'] = True
                        atom[idx] = temp
                else:
                    return None
                    

                final_prop = dict()
                final_prop["str"] = atom["name"] + \
                    "(?" + atom['var1']['name'] + ",?" + atom['var2']['name'] + ")"
                final_prop["obj"] = atom
                atoms.append(final_prop)

        print("\nDo you want to add another atom?")
        option = input("y/[n]: ")
        if not (option == "y" or option == "Y"):
            running = False

    final_query = dict()
    final_query["obj"] = atoms
    dist_str = ""
    for var in distinguished:
        dist_str += var
        dist_str += ",?"
    dist_str = dist_str[:-2]
    final_query["str"] = "q(?" + dist_str + ") :- "
    for i in range(len(atoms)):
        final_query["str"] += atoms[i]["str"]
        if i < ((len(atoms)) - 1):
            final_query["str"] += "^"

    return final_query

def ready_to_run(ontology):
    if ontology == None:
        return False
    return True

def create_test_environment(project_name):
    clear()
    env_path = "testcases/" + project_name
    if os.path.isdir(env_path):
        print("This test case already exists. You will not overwrite a model configuration, i.e, if the same model, dimension and epoch number.\n If you want a new test environment, please change the identifier.")
        press_any_key()
    else:
        os.makedirs(env_path)
        print("Test environment successfully created.")
        press_any_key()

def print_kg_results(queries):
    for q in queries:
        for atom in q:
            if atom['atom']['type'] == "concept":
                print(atom['atom']['name']+"(%s):" % (atom['atom']['var1']['name']))
                if not atom['returned_entities']:
                    print("Empty lookup.\n")
                else:

                    for i in range(len(atom['returned_entities'])):
                        if i < 20:
                            print(atom['returned_entities'][i][0])
                        if i == 20:
                            print("[...]")
                        if i == len(atom['returned_entities']) - 1:
                            print("Total number of entities: %i\n" % (len(atom['returned_entities'])))

                   
            if atom['atom']['type'] == "role":
                print(atom['atom']['name']+"(%s,%s):" % (atom['atom']['var1']['name'],atom['atom']['var2']['name']))
                if not atom['returned_entities']:
                    print("Empty lookup.\n")
                else:
                    if atom['atom']['var1']['is_bound']:
                        for i in range(len(atom['returned_entities'][0])):
                            if i < 20:
                                print(atom['returned_entities'][0][i])
                            if i == 20:
                                print("[...]")
                            if i == len(atom['returned_entities'][0]) - 1:
                                print("Total number of entities: %i\n" % (len(atom['returned_entities'][0])))

                    if atom['atom']['var2']['is_bound']:
                        for i in range(len(atom['returned_entities'][1])):
                            if i < 20:
                                print(atom['returned_entities'][1][i])
                            if i == 20:
                                print("[...]")
                            if i == len(atom['returned_entities'][1]) - 1:
                                print("Total number of entities: %i\n" % (len(atom['returned_entities'][1])))

def print_kge_results(queries):
    return None

def write_results_to_file(queries, original_query, transductive_models, project_name, dataset):
    path = "testcases/"+project_name+"/"
    file_name = path + original_query + ".txt"
    # File separation
    f = open(file_name, "w")
    for q in queries:
        for atom in q:
            if atom['atom']['type'] == "concept":
                f.write(atom['atom']['name']+"(%s):\n" % (atom['atom']['var1']['name']))
                if not atom['returned_entities']:
                    f.write("Empty lookup.\n")
                else:
                    for i in range(len(atom['returned_entities'])): 
                        f.write(atom['returned_entities'][i][0]+"\n")                          
                        if i == len(atom['returned_entities']) - 1:
                            f.write("Total number of entities: %i\n" % (len(atom['returned_entities'])))
                
            if atom['atom']['type'] == "role":
                f.write(atom['atom']['name']+"(%s,%s):\n" % (atom['atom']['var1']['name'],atom['atom']['var2']['name']))
                if not atom['returned_entities']:
                    f.write("Empty lookup.\n")
                else:
                    if atom['atom']['var1']['is_bound']:
                        for i in range(len(atom['returned_entities'][0])): 
                            f.write(atom['returned_entities'][0][i]+"\n")  
                            if i == len(atom['returned_entities'][0]) - 1:
                                f.write("Total number of entities: %i\n" % (len(atom['returned_entities'][0])))
                    if atom['atom']['var2']['is_bound']:
                        for i in range(len(atom['returned_entities'][1])):  
                            f.write(atom['returned_entities'][1][i]+"\n")           
                            if i == len(atom['returned_entities'][1]) - 1:
                                f.write("Total number of entities: %i\n" % (len(atom['returned_entities'][1])))
        f.write("\n")
    f.close()   

    #File merged
    uniques = list()
    file_name = path + original_query + "_entailed.txt"
    for q in queries:
        for atom in q:
            if atom['atom']['type'] == "concept":
                if atom['returned_entities']:
                    for i in range(len(atom['returned_entities'])): 
                        if atom['returned_entities'][i][0] not in uniques:
                            uniques.append(atom['returned_entities'][i][0])                          
            if atom['atom']['type'] == "role":
                if atom['returned_entities']:
                    if atom['atom']['var1']['is_bound']:
                        for i in range(len(atom['returned_entities'][0])): 
                            if atom['returned_entities'][0][i] not in uniques:
                                uniques.append(atom['returned_entities'][0][i])                             
                    if atom['atom']['var2']['is_bound']:
                        for i in range(len(atom['returned_entities'][1])):  
                            if atom['returned_entities'][1][i] not in uniques:
                                uniques.append(atom['returned_entities'][1][i])     
    f = open(file_name, "w")
    f.write("Total number of entities: %i\n" % (len(uniques)))
    f.write("Original_query: %s\n" % (original_query))
    for entity in uniques:
        f.write(entity+"\n")
    f.close()

def write_results_to_file_kge(entity_df, original_query, project_name):
    path = "testcases/"+project_name+"/"
    keys = entity_df.keys()
    for key in keys:
        file_name = path + original_query + "_kge_"+str(key)+".txt"
        entity_df[key].to_csv(path_or_buf=file_name)

def update_prediction_pickle(structure, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(structure, file, protocol=pickle.HIGHEST_PROTOCOL)

def t_norm(a, b):
    return a*b

def tco_norm(a, b):
    return max(a, b)