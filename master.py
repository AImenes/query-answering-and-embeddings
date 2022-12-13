#Import system based
import os
import pathlib
from time import sleep

#Import PerfectRef and OwlReady2
import perfectref_v1 as pr
from owlready2 import get_ontology


#Import embedding related
from torch.cuda import is_available
from pykeen.models import *
from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
from pykeen.models.uncertainty import predict_hrt_uncertain
from pykeen import predict

# SETTINGS
tbox_import_file = "family_ontology.owl"
tbox_ontology = None
query = "q(?x) :- Sibling(?x)"
dataset = "family"
t_box_path = "dataset/"+dataset+"/tbox/"+tbox_import_file
a_box_path = "dataset/"+dataset+"/abox/transductive/"
project_name = "001"
entailed_queries = None
transductive_models = ["TransE", "BoxE", "RotatE"]
inductive_models = ["InductiveNodePieceGNN"]


# Embedding-specific
transductive_train_set = TriplesFactory.from_path(a_box_path + "train.txt")
transductive_valid_set = TriplesFactory.from_path(a_box_path + "validation.txt", entity_to_id=transductive_train_set.entity_to_id, relation_to_id=transductive_train_set.relation_to_id)
transductive_test_set = TriplesFactory.from_path(a_box_path + "test.txt", entity_to_id=transductive_train_set.entity_to_id, relation_to_id=transductive_train_set.relation_to_id)

# HYPERPARAMETERS
RANDOM_SEED = 41279374

# Methods
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

def change_dataset():
    if dataset == "family":
        return "nell"
    else:
        return "family"

def load_ontology(path):
    ontology = get_ontology(t_box_path).load()
    return ontology

def show_tbox(tbox):
    print("CLASSES:\n", list(tbox.classes()))
    print("\n\nPROPERTIES:\n1",list(tbox.properties()))

def enter_query(ontology):
    if ontology == None:
        print("Please import a tbox before using the query wizard.")
        press_any_key()
        return None
    temp = input("Enter distinguished variable name(default = x): ")
    if temp == "":
        distinguished = "x"
    else:
        distinguished = temp

    classes = list()
    properties = list()
    for entity in list(ontology.classes()):
        classes.append(entity.name)

    for prop in list(ontology.properties()):
        properties.append(prop.name)

    atoms = list()
    running = True
    while running:

        atom = input("Enter atom name:")

        #Classes
        if (atom in classes):
            print("Use distinguished variable %s?" % (distinguished))
            option = input("[y]/n: ")
            if (option == "" or option == "y" or option == "Y"):
                completed_atom = atom + "(?" + distinguished + ")"
            else:
                new_variable = input("Input other variable (If left blank, will add unbound variable holder) : ")
                if new_variable == "":
                    completed_atom = atom + "(?_)"
                else:
                    completed_atom = atom + "(?" + new_variable + ")"
            
            atoms.append(completed_atom)

        #Properties 
        elif (atom in properties):
            print("You have selected a property, and need two variables.\n\n")
            is_used = False
            variables = [None, None]
            for i in range(len(variables)):
                print("Variable %d: " % (i+1))
                if not is_used:
                    print("Use distinguished variable %s for variable %d?" % (distinguished, i+1))
                    option = input("[y]/n: ") 
                    if (option == "" or option == "y" or option == "Y"):
                        is_used = True
                        variables[i] = distinguished
                    else:
                        new_input = input("Enter variable name. Leave empty if unbound: ")
                        variables[i] = new_input
                        if variables[i] == "":
                            variables[i] = "_"
                else:
                    print("Enter variable %d. " % (i+1))
                    variables[i] = input("Leave empty if unbound: ")
                    if variables[i] == "":
                            variables[i] = "_"
            final_prop = atom + "(?" + variables[0] + ",?" + variables[1] + ")"
            atoms.append(final_prop)

        else:
            print("Error. Atom not found. Remember to be case-sensitive.")
            break

        print("Do you want to add another atom?")
        option = input("y/n: ")
        if not (option == "y" or option == "Y"):
            running = False
    
        
    final_query = "q(?" + distinguished + ") :- "
    for i in range(len(atoms)):
            final_query += atoms[i]
            if i < ((len(atoms)) - 1):
                final_query += "^"
    
    return final_query

def ready_to_run():
    if tbox_ontology == None:
        return False
    return True

def create_test_environment():
    clear()
    env_path = "testcases/" + project_name
    if os.path.isdir(env_path):
        print("This test case already exists. You will not overwrite a model configuration, i.e, if the same model, dimension and epoch number.\n If you want a new test environment, please change the identifier.")
        press_any_key()
    else:
        os.makedirs(env_path)
        print("Test environment successfully created.")
        press_any_key()

def precompletion():
    
    #step 1 - train
    for current_model in transductive_models:
        dim = int(input("Enter dimension for model %s: " % (current_model)))
        epochs = int(input("Enter number of epochs for model %s: " % (current_model)))
        current_config = "testcases/" + project_name + "/precompletion/dataset:" + dataset + "_model:" + current_model + "_dim:" + str(dim) + "_epoch:" + str(epochs)
        
        if os.path.exists(current_config):
            print("Configuration already trained. Edit configuration or delete model directory.")
        else:
            results = pipeline(
                training_loop='sLCWA', 
                training = transductive_train_set,
                testing = transductive_test_set,
                random_seed=RANDOM_SEED,
                model=current_model,
                model_kwargs=dict(embedding_dim=dim),
                epochs=epochs,
                device=get_device(),
            )

            results.save_to_directory(current_config)    

        # step 2 - predict top k new queries
        partial_dataset = predict.PartiallyRestrictedPredictionDataset()
        consumer = predict.TopKScoreConsumer(k=3)
        predict.consume_scores(results.model, da)

    # step 3 - add the results to the original knowledge graph

    # step 4 - query the graph

def new_approach():
    # step 1 - train

    # if single-atom queries

        # step 2- Predict on atom, top k predictions

        # step 3- Add new triples to KG, and repeat until no further changes

    # Else (multiple atoms)

        # iterate through atoms and perform as with single atoms. At last, use a norm to get a conjuncted score.
    return 0

def query_embedding():
    # have query

    # first precompletion, then our approach

    # 
    
    return 0


#menu
menu = {}
menu['1']="Create new test case or extend to existing one"
menu['2']="Change dataset"
menu['3']="Load TBox"
menu['4']="Load ABox"
menu['5']="Change query"
menu['6']="Show entities and properties in TBox"
menu['7']="Run PerfectRef (entail queries)"
menu['8']="Run Pre-completion"
menu['9']="Run our approach"
menu['10']="Compare results"
menu['0']="Exit"

while True:
    clear()
    options=menu.keys()
    sorted(options)
    print("\n ----------------------------------------------------")
    print(" |          QUERY ANSWERING AND EMBEDDINGS           ")
    print(" ----------------------------------------------------")
    print(" | Test case identifier:\t%s" % (project_name))
    print(" ----------------------------------------------------")
    print(" | QUERIES ")
    print(" | Current TBox file:\t\t%s" % (tbox_import_file))
    print(" | Loaded TBox state:\t\t%s" % (tbox_ontology != None))
    print(" | Current CQ:\t\t\t%s\n" % (query))
    print(" | EMBEDDINGS")
    print(" | Current dataset:\t\t%s" % (dataset))
    print(" | Current transductive models:\t", end="")
    if not transductive_models == None:
        for a in transductive_models:   
            print("%s, " % (a), end= " ")  
    print("\n | Current inductive models:\t", end="")
    if not inductive_models == None:
        for a in inductive_models:   
            print("%s, " % (a), end= " ")  
    print("\n | Is GPU available (Cuda):\t%s\n" % (is_available()))
    print(" | Ready to run? \t\t%s" % (ready_to_run()))
    print(" ----------------------------------------------------")
    print(" | PERFECTREF Entailed triples:")
    if not entailed_queries == None:
        for a in entailed_queries['entailed']:   
            print(" | %s" % (a))
            


    print("\n --------------------- MENU -------------------------")

    for entry in options: 
        print(" | "+entry+":\t", menu[entry])

    selection=input("\nPlease Select an option [0-9]: ") 
    if selection =='0':     
        print("\nProgram exited successfully.\n")
        break
    elif selection == '1': 
        project_name = input("Enter test case identifier (folder name)  \n[001]: ")
        if project_name == "":
            project_name = "001"
    elif selection == '2':
        if dataset == "nell":
            dataset = "family"
            tbox_import_file = "family_ontology.owl"
            t_box_path = "dataset/"+dataset+"/tbox/"+tbox_import_file
            a_box_path = "dataset/"+dataset+"/abox/transductive/"
        else:
            dataset = "nell"
            tbox_import_file = "nell.owl"
            t_box_path = "dataset/"+dataset+"/tbox/"+tbox_import_file
            a_box_path = "dataset/"+dataset+"/abox/transductive/"
    elif selection == '3': 
        tbox_ontology = load_ontology(t_box_path)
        print("\n\nTBox loaded successfully.")
        press_any_key()
    elif selection == '4':
        print("To be implemented")
    elif selection == '5':
        opt = input("Press [0] for using the query wizard or press [1] for entering manually [0/1]: ")
        if opt == "0":
            query = enter_query(tbox_ontology)
        elif opt == "1":
            query = input("Manually input the query (on this format: q(?x) :- atom(?x)^another(?_,?x)): ")
        else:
            print("You have entered something illegal. No changes were made.")
            press_any_key()
    elif selection == '6':
        clear()
        if not tbox_ontology == None:
            show_tbox(tbox_ontology)
        else:
            print("Please import a tbox first.")
        press_any_key()
    elif selection == '7':
        entailed_queries = pr.parse_output(query, pr.get_entailed_queries(t_box_path, query))
    elif selection == '8':
        precompletion()
    elif selection == '9':
        new_approach()    
    elif selection == '10':
        print("To be implemented")
    else: 
      print("Unknown Option Selected! Select a number [0-10].")
      press_any_key()
