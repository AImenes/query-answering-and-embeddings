# Import system based
import os
import pandas as pd
import json
import pickle

# Import PerfectRef and OwlReady2
import perfectref_v1 as pr

# Import embedding related
from pykeen.triples import TriplesFactory

#import own modules
from query_generator import *
from query_parser import *
from predict import *
from utilities import *
from kglookup import *
from models import *

# Pandas settings
pd.set_option('display.max_rows', 100)

def main():
    # SETTINGS

    # HYPERPARAMETERS
    RANDOM_SEED = 41279374
    tbox_import_file = "dbpedia15k.owl"
    query = None
    dataset = "dbpedia15k"
    t_box_path = "dataset/"+dataset+"/tbox/"+tbox_import_file
    a_box_path = "dataset/"+dataset+"/abox/transductive/"
    tbox_ontology = load_ontology(t_box_path)
    project_name = "001"
    transductive_models = ["TransE", "BoxE", "RotatE", "DistMult", "CompGCN"]
    inductive_models = ["InductiveNodePieceGNN"]
    tf = TriplesFactory.from_path(a_box_path + "all.txt")
    train, valid, test = tf.split([0.8, 0.1, 0.1], random_state=RANDOM_SEED, method="cleanup")
    current_model = None
    current_model_params = {'selected_model_name': None, 'dim': None, 'epoch': None}
    entailed_queries = None
    parsed_entailed_queries = None
    parsed_generated_queries = None
    parsed_entailed_queries_subset = None
    received_entities_kg_queries = None
    received_entities_kge_queries = None
    atom_mapping = [None]
    number_of_queries_per_structure = None
    queries_from_generation = None

    use_storage_file = False

    base_cases_path  = "testcases/" + project_name + "/base_cases.pickle"
    
    if os.path.exists(base_cases_path):
        with open(base_cases_path, 'rb') as file:
            base_cases = pickle.load(file)
    else:
        base_cases = list()

    # menu
    menu = {}
    menu['1'] = "Create new test case or extend to existing one"
    menu['2'] = "Change dataset"
    menu['3'] = "Load TBox"
    menu['4'] = "Generate queries"
    menu['5'] = "Import set of queries (JSON)"
    menu['6'] = "Change query"
    menu['7'] = "Show entities and properties in TBox"
    menu['8'] = "Run PerfectRef (entail queries)"
    menu['9'] = "Select queries to ask"
    menu['10'] = "Train models"
    menu['11'] = "Load model"
    menu['12'] = "Query KG with original query"
    menu['13'] = "Query KG with a subset of entailed queries"
    menu['14'] = "Query KG with all entailed queries (expensive)"
    menu['15'] = "Query KGE with original query"
    menu['16'] = "Query KGE with a subset of entailed queries"
    menu['17'] = "Query KGE with all entailed queries (expensive)"
    menu['18'] = "Compare results"
    menu['0'] = "Exit"

    while True:
        clear()
        options = menu.keys()
        sorted(options)
        print("\n ----------------------------------------------------")
        print(" |          QUERY ANSWERING AND EMBEDDINGS           ")
        print(" ----------------------------------------------------")
        print(" | Test case identifier:\t%s" % (project_name))
        print(" ----------------------------------------------------")
        print(" | QUERIES ")
        print(" | Current TBox file:\t\t%s" % (tbox_import_file))
        print(" | Loaded TBox state:\t\t%s" % (tbox_ontology != None))
        print(" | Current CQ:\t\t\t%s\n |" % (query))
        print(" | EMBEDDINGS")
        print(" | Current dataset:\t\t%s" % (dataset))
        print(" | Current transductive models:\t", end="")
        if not transductive_models == None:
            for a in transductive_models:
                print("%s, " % (a), end=" ")
        print("\n | Current inductive models:\t", end="")
        if not inductive_models == None:
            for a in inductive_models:
                print("%s, " % (a), end=" ")
        print("\n | Is GPU available (Cuda):\t%s\n |" % (is_available()))
        print(" | Ready to run? \t\t%s" % (ready_to_run(tbox_ontology)))
        print(" ----------------------------------------------------")
        print(" | PERFECTREF Entailed triples:")
        if not entailed_queries == None:
            for i in range(len(entailed_queries_str['entailed'])):
                if i < 20:
                    print(" | %s" % (entailed_queries_str['entailed'][i]))
                elif i == 20:
                    print(" | ...\n | Total number of queries: %i\n |" % (len(entailed_queries_str['entailed'])))
                else:
                    pass

        print(" --------------------- MENU -------------------------")

        for entry in options:
            print(" | "+entry+":\t", menu[entry])

        selection = input("\nPlease Select an option [0-17]: ")
        if selection == '0':
            print("\nProgram exited successfully.\n")
            break
        elif selection == '1':
            project_name = input(
                "Enter test case identifier (folder name)  \n[001]: ")
            if project_name == "":
                project_name = "001"

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
                
                atom_mapping = pd.read_csv(a_box_path + "atomid2label.txt", sep="\t", header=None)
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
            tbox_ontology = load_ontology(t_box_path)
            print("\n\nTBox loaded successfully.")
            press_any_key()
        elif selection == '4':
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
        elif selection == '5':
            number_of_queries_per_structure = int(input("Please input the number of queries to generate per query type (1-100): "))
            query_path = "testcases/" + project_name + "/queries/" + dataset + "/queries_k-" + str(number_of_queries_per_structure) + ".txt"
            parsed_generated_queries = dict()
            with open(query_path) as json_file:
                queries_from_generation = json.load(json_file)
            for key in queries_from_generation.keys():
                parsed_generated_queries[key] = list()
                for q in queries_from_generation[key]:
                    parsed_generated_queries[key].append(query_structure_parsing(q, key, tbox_ontology))
            
            #do KG-lookup
            print("Successfully imported! \n\nPerforming KG-lookup on imported queries ...")
            parsed_generated_queries = kg_lookup(parsed_generated_queries, dataset, a_box_path, tf)
            print("\nKG-lookup finished, and answers stored with query variable.\n")
            press_any_key()

        elif selection == '6':
            query_object = enter_query(tbox_ontology)
            if query_object is not None:
                query = query_object["str"]
        elif selection == '7':
            clear()
            if not tbox_ontology == None:
                show_tbox(tbox_ontology)

                if dataset == "family":
                    try:
                        print(atom_mapping)
                    except:
                        print("Mapping file missing.")
            else:
                print("Please import a tbox first.")
            press_any_key()
        elif selection == '8':
            entailed_queries = pr.get_entailed_queries(t_box_path, query)
            entailed_queries_str = pr.parse_output(query, entailed_queries)
        elif selection == '9':
            if entailed_queries == None:
                print("Please run PerfectRef first.")
            else:
                for i in range(len(entailed_queries_str['entailed'])):
                    print("[%i]:\t%s" % (i, entailed_queries_str['entailed'][i]))
                options = input("\nEnter wanted queries by commaseparation (ex: '1,3,4'): ")
                query_ids = options.split(",")
                query_ids = [int(s) for s in query_ids]
                queries_to_be_parsed = list()
                for id in query_ids:
                    queries_to_be_parsed.append(entailed_queries[id])
            parsed_entailed_queries_subset = parse_entailed_queries(queries_to_be_parsed)
                
        elif selection == '10':
            training(transductive_models, project_name,
                     dataset, train, valid, test)
        elif selection == '11':
            current_model, current_model_params = load_model(tf, train, valid, test, project_name, dataset)
        elif selection == '12':
            if query is None:
                print("Please enter a query first.")
                press_any_key()
            else:
                received_entities_kg_queries = list()
                list_of_atoms = list()
                for atom in query_object['obj']:
                    temp = dict()
                    #temp['str'] = query_object['str']
                    temp['atom'] = atom['obj']
                    if atom['obj']['type'] == "concept":
                        temp['returned_entities']=query_graph_concepts(atom['obj'], dataset, a_box_path)
                    if atom['obj']['type'] == "role":
                        temp['returned_entities']=query_graph_roles(atom['obj'], tf)
                    list_of_atoms.append(temp)
                received_entities_kg_queries.append(list_of_atoms)
                print_kg_results(received_entities_kg_queries)
                write_results_to_file(received_entities_kg_queries, query, transductive_models, project_name, dataset)
                press_any_key()

        elif selection == '13':
            if parsed_entailed_queries_subset is None:
                print("Please run PerfectRef first (7), and select subset (8).")
                press_any_key()
            else:
                received_entities_kg_queries = list()
                for q in parsed_entailed_queries_subset:
                    list_of_atoms = list()
                    for atom in q['obj']:
                        temp = dict()
                        temp['atom'] = atom['obj']
                        if atom['obj']['type'] == "concept":
                            temp['returned_entities'] = query_graph_concepts(atom['obj'], tbox_ontology, a_box_path)
                        if atom['obj']['type'] == "role":
                            temp['returned_entities'] = query_graph_roles(atom['obj'], tf)
                        list_of_atoms.append(temp)
                    received_entities_kg_queries.append(list_of_atoms)
                print_kg_results(received_entities_kg_queries)
                write_results_to_file(received_entities_kg_queries, query, transductive_models, project_name, dataset)
                press_any_key()

        elif selection == '14':
            if entailed_queries is None:
                print("Please run PerfectRef first (7).")
            else:
                parsed_entailed_queries = parse_entailed_queries(entailed_queries)
                received_entities_kg_queries = list()
                for q in parsed_entailed_queries:
                    list_of_atoms = list()
                    for atom in q['obj']:
                        temp = dict()
                        temp['atom'] = atom['obj']
                        if atom['obj']['type'] == "concept":
                            temp['returned_entities'] = query_graph_concepts(atom['obj'], tbox_ontology, a_box_path)
                        if atom['obj']['type'] == "role":
                            temp['returned_entities'] = query_graph_roles(atom['obj'], tf)
                        list_of_atoms.append(temp)
                    received_entities_kg_queries.append(list_of_atoms)
                print_kg_results(received_entities_kg_queries)
                write_results_to_file(received_entities_kg_queries, query, transductive_models, project_name, dataset)
                press_any_key()
        elif selection == '15':
            if current_model is None:
                print("Please load a model (10) you want to utilize for prediction.")
                press_any_key()
            else:
                #received_entities_kge_queries = new_approach(current_model,100, tf, train, valid, test, [query_object], tbox_ontology, a_box_path)
                #entities_df = combine_scores_old(received_entities_kge_queries, query, transductive_models, project_name, dataset)      
                write_results_to_file_kge(entities_df, query, project_name)
        elif selection == '16':
            if current_model is None:
                print("Please load a model (10) you want to utilize for prediction.")
                press_any_key()
            elif entailed_queries is None:
                print("\nRun PerfectRef (7) first.")
                press_any_key()
            elif parsed_entailed_queries_subset is None:
                print("\nRun Select a subset of queries (8) first.")
                press_any_key()
            else:
                #received_entities_kge_queries = new_approach(current_model,100, tf, train, valid, test, parsed_entailed_queries_subset, tbox_ontology, a_box_path)
                #entities_df = combine_scores_old(received_entities_kge_queries, query, transductive_models, project_name, dataset)      
                write_results_to_file_kge(entities_df, query, project_name)
        
        elif selection == '17':
            if entailed_queries is None:
                print("\nRun PerfectRef first.")
                press_any_key()
            elif current_model is None:
                print("\nPlease load a model (10) you want to utilize for prediction.")
                press_any_key()
            else:
                parsed_entailed_queries = parse_entailed_queries(entailed_queries)
                #received_entities_kge_queries = new_approach(current_model,100, tf, train, valid, test, parsed_entailed_queries, tbox_ontology, a_box_path)
                #entities_df = combine_scores_old(received_entities_kge_queries, query, transductive_models, project_name, dataset)      
                #write_results_to_file_kge(entities_df, query, project_name)

        elif selection == '18':
            if parsed_generated_queries is None or current_model is None:
                print("Import a set of queries first (option 5) and a model (option 11).")
                press_any_key()
            else:
                #filename
                pth = "testcases/" + project_name + "/queries/" + dataset + "/"
                filename = "queries" + "_k-" + str(number_of_queries_per_structure) + "_parsed_and_rewritten.pickle"
                full_pth = pth + filename
                
                #Reformulate
                parsed_generated_queries = query_reformulate(parsed_generated_queries, full_pth, t_box_path) 

                #Predict
                parsed_generated_queries = predict_parsed_queries(parsed_generated_queries,base_cases, base_cases_path, dataset, current_model, current_model_params, 100, tf, train, valid, test, tbox_ontology, a_box_path, use_storage_file)

    
                #Combine scores
                for query_structure in parsed_generated_queries.keys():

                    #for each query in that structure
                    for query_dict in parsed_generated_queries[query_structure]:
                        query_dict = combine_scores(query_dict)

                print("ohboi")

        else:
            print("Unknown Option Selected! Select a number [0-18].")
            press_any_key()


if __name__ == "__main__":
    main()
else:
    print("Not opened directly.")
