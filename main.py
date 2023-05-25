# Import system based
import os
import pandas as pd
import json
import pickle
from pathlib import Path

# Import embedding related
from pykeen.triples import TriplesFactory

#import custom modules
from query_generator import *
from query_parser import *
from predict import *
from utilities import *
from kglookup import *
from models import *

def main():
    """
    An interactive console application for generating, parsing, and predicting queries with knowledge graph embeddings with PerfectRef implemented.
    Application is made together with my Masters Thesis, Combining Query Rewriting with Complex Query Answering.
    
    Code is written by 
    Anders Imenes
    aimenes@icloud.com
    """

    # Define hyperparameters
    RANDOM_SEED = 41279374  

    #PerfectRef upper rewriting limit. Default = 100
    rewriting_upper_limit = 100

    #For Hits@N
    n = 3
    
    #Save top k predictions. This highly changes running time. Default = 100. 
    k = 100 
    #########################################################################
    ################          DEFAULT LOADING VALUES       ##################
    tbox_import_file = "dbpedia15k.owl"
    dataset = "dbpedia15k"
    t_box_path = "dataset/"+dataset+"/tbox/"+tbox_import_file
    a_box_path = "dataset/"+dataset+"/abox/transductive/"
    tbox_ontology = load_ontology(t_box_path)
    project_name = "002"
    tf = TriplesFactory.from_path(a_box_path + "all.txt")
    train, valid, test = tf.split([0.8, 0.1, 0.1], random_state=RANDOM_SEED, method="cleanup")
    current_model = None
    current_model_params = {'selected_model_name': None, 'dim': None, 'epoch': None}
    parsed_generated_queries = None
    number_of_queries_per_structure = None
    queries_from_generation = None
    enable_online_lookup = True
    base_cases_path  = "testcases/" + project_name + "/base_cases.pickle"
    
    #Load previous base predictions
    if os.path.exists(base_cases_path):
        with open(base_cases_path, 'rb') as file:
            base_cases = pickle.load(file)
    else:
        base_cases = list()

    #Create test environment
    if not os.path.exists("testcases/" + project_name):
        os.mkdir("testcases/" + project_name)
        os.mkdir("testcases/" + project_name + "/models")
        os.mkdir("testcases/" + project_name + "/queries")
        os.mkdir("testcases/" + project_name + "/queries/family")
        os.mkdir("testcases/" + project_name + "/queries/dbpedia15k")

    #########################################################################

    ##########################     Menu-loop    ############################
    menu = {
            '1': "Create/change test environment",
            '2': "Change dataset\t\t(Family / DBPedia15k)",
            '3': "Generate queries",
            '4': "Import set of queries\t(JSON-file)",
            '5': "Train model",
            '6': "Load model",
            '7': "Run query answering\t(PerfectRef incl.)",
            '0': "Exit"
        }

    while True:
        clear()
        options = menu.keys()
        sorted(options)
        print("\n ----------------------------------------------------")
        print(" |           QUERY REWRITING and KGEs v2.0          |")
        print(" ----------------------------------------------------")
        print("   Test case identifier:\t%s" % (project_name))
        print(" ----------------------------------------------------")
        print("   QUERIES ")
        print("   Current TBox file:\t\t%s" % (tbox_import_file))
        print("   Loaded TBox state:\t\t%s" % (tbox_ontology != None))
        print("   Queries loaded:\t\t%s\n  " % (parsed_generated_queries != None))
        print("   EMBEDDINGS")
        print("   Current dataset:\t\t%s" % (dataset))
        print("   Is GPU available (Cuda):\t%s\n  " % (is_available()))
        print("   Ready to run? \t\t%s" % (parsed_generated_queries != None and bool(parsed_generated_queries) and is_available() and tbox_ontology != None))
        print("\n --------------------- MENU -------------------------")

        for entry in options:
            print(" "+entry+":\t", menu[entry])

        selection = input("\nPlease select an option [0-7]: ")
        if selection == '0':
            print("\nProgram exited successfully.\n")
            break
        elif selection == '1':
            project_name = input(
                "Enter test case identifier (folder name)  \n[001]: ")
            if project_name == "":
                project_name = "001"

            if not os.path.exists("testcases/" + project_name):
                os.mkdir("testcases/" + project_name)
                os.mkdir("testcases/" + project_name + "/models")
                os.mkdir("testcases/" + project_name + "/queries")
                os.mkdir("testcases/" + project_name + "/queries/family")
                os.mkdir("testcases/" + project_name + "/queries/dbpedia15k")

        elif selection == '2':
            if dataset == "dbpedia15k":
                dataset = "family"
                tbox_import_file = "family_wikidata.owl"
                t_box_path = "dataset/"+dataset+"/tbox/"+tbox_import_file
                a_box_path = "dataset/"+dataset+"/abox/transductive/"
                tbox_ontology = load_ontology(t_box_path)
                tf = TriplesFactory.from_path(a_box_path + "all.txt", create_inverse_triples=False)
                train, valid, test = tf.split(
                    [0.8, 0.1, 0.1], random_state=RANDOM_SEED, method="cleanup")       
            else:
                dataset = "dbpedia15k"
                tbox_import_file = "dbpedia15k.owl"
                t_box_path = "dataset/"+dataset+"/tbox/"+tbox_import_file
                a_box_path = "dataset/"+dataset+"/abox/transductive/"
                tbox_ontology = load_ontology(t_box_path)
                ent_to_id = dict()
                rel_to_id = dict()

                with open(a_box_path + "entity2id.txt") as f:
                    for line in f:
                        (val, key) = line.split()
                        ent_to_id[val] = int(key)

                with open(a_box_path + "relation2id.txt") as f:
                    for line in f:
                        (val, key) = line.split()
                        rel_to_id[val] = int(key)
                tf = TriplesFactory.from_path(
                    a_box_path + "all.txt", entity_to_id=ent_to_id, relation_to_id=rel_to_id, create_inverse_triples=False)
                train, valid, test = tf.split(
                    [0.8, 0.1, 0.1], random_state=RANDOM_SEED, method="cleanup")
        elif selection == '3':
            pth = "testcases/" + project_name + "/queries/" + dataset + "/"
            filename = "queries" + "_k-" + str(number_of_queries_per_structure) + ".txt"
            full_pth = pth + filename
            number_of_queries_per_structure = int(input("Please input the number of queries to generate per query type (1-100): "))
            if not os.path.exists(full_pth):
                query_gen(project_name, number_of_queries_per_structure)
                print("Queries successfully created, and can be found in the testcases-folder.")
            else:
                print("A query configuration for k = " + str(number_of_queries_per_structure) + " already exists.")
            press_any_key()
        elif selection == '4':
            parsed_generated_queries = dict()
            unloaded = True

            while unloaded:
                number_of_queries_per_structure = int(input("Please input the number of queries to generate per query type (1-100), or press 0 to cancel: "))
                query_path = "testcases/" + project_name + "/queries/" + dataset + "/queries_k-" + str(number_of_queries_per_structure) + ".txt"
                if number_of_queries_per_structure == 0:
                    unloaded = False
                else:
                    try:
                        with open(query_path) as json_file:
                            queries_from_generation = json.load(json_file)
                        unloaded = False
                    except:
                        print("Doesnt exist. Try again.")
            if number_of_queries_per_structure != 0:
                for key in queries_from_generation.keys():
                    parsed_generated_queries[key] = list()
                    for q in queries_from_generation[key]:
                        parsed_generated_queries[key].append(query_structure_parsing(q, key, tbox_ontology))
                
                #do local original KG-lookup
                pth = "testcases/" + project_name + "/queries/" + dataset + "/"
                filename = "queries" + "_k-" + str(number_of_queries_per_structure) + "_parsed_and_rewritten.pickle"
                full_pth = pth + filename
                print("Successfully imported! \n\nPerforming KG-lookup on imported queries ...")
                if not os.path.exists(full_pth):
                    parsed_generated_queries = kg_lookup(parsed_generated_queries, dataset, a_box_path, tf)
                else:
                    print("\n Reformulation already exists. Loaded pickle for this configuration. Delete or rename the pickle file if you want to redo the reformulation. \n")
                    with open(full_pth, 'rb') as handle:
                        parsed_generated_queries = pickle.load(handle)
                print("\nLocal KG-lookup finished, and answers stored with query variable.\n")
                
                result_path = Path(f"testcases/{project_name}/queries/{dataset}/k-{number_of_queries_per_structure}/")
                result_path.mkdir(parents=True, exist_ok=True)
                result_every_structure = Path(f"testcases/{project_name}/queries/{dataset}/k-{number_of_queries_per_structure}/every_structure/")
                result_every_structure.mkdir(parents=True, exist_ok=True)
                press_any_key()
                
        elif selection == '5':
            training(RANDOM_SEED, project_name, dataset, train, valid, test)
        elif selection == '6':
            current_model, current_model_params = load_model(project_name, dataset)

        elif selection == '7':
            if parsed_generated_queries is None or current_model is None:
                print("Import a set of queries first (option 5) and a model (option 11).")
                press_any_key()
            else:
                #filename
                pth = "testcases/" + project_name + "/queries/" + dataset + "/"
                filename = "queries" + "_k-" + str(number_of_queries_per_structure) + "_parsed_and_rewritten.pickle"
                full_pth = pth + filename
                
                #Reformulate
                parsed_generated_queries = query_reformulate(parsed_generated_queries, rewriting_upper_limit, full_pth, t_box_path) 

                #Reformulation KG Lookup
                parsed_generated_queries = kg_lookup_rewriting(parsed_generated_queries, dataset, a_box_path, tf, full_pth)

                #Predict
                predict_parsed_queries(parsed_generated_queries,base_cases, base_cases_path, enable_online_lookup, dataset, current_model, current_model_params, k, tf, train, valid, test, tbox_ontology, a_box_path, result_path, n)


        else:
            print("Unknown option selected! Please select a acceptable number [0-7].")
            press_any_key()


if __name__ == "__main__":
    main()
else:
    print("Not opened directly.")
