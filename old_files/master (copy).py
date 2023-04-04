# Import system based
import os
import pathlib
from time import sleep
import numpy as np
import pandas as pd
import functools as ft
import torch
import json
import pickle
import copy
from scipy.special import expit

# Import PerfectRef and OwlReady2
import perfectref_v1 as pr
from perfectref_v1 import Query, QueryBody
from perfectref_v1 import AtomParser, AtomConcept, AtomRole, AtomConstant
from perfectref_v1 import Variable, Constant
from owlready2 import get_ontology
import query_generator
import query_parser


# Import embedding related
from torch import load
from torch.cuda import is_available
from pykeen.models import *
from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
from pykeen.models.uncertainty import predict_hrt_uncertain
from pykeen import predict


# HYPERPARAMETERS
RANDOM_SEED = 41279374

# Pandas settings
pd.set_option('display.max_rows', 100)


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

def training(transductive_models, project_name, dataset, train, valid, test):
    # step 1 - train
    epochs = [16,20,24]
    dims = [50,128,192]
    press_any_key()
    #for dim in dims:
    #    for epoch in epochs:
    for current_model in transductive_models:
        dim = int(input("Enter dimension for model %s: " % (current_model)))
        epoch = int(input("Enter number of epochs for model %s: " % (current_model)))
        current_config = "testcases/" + project_name + "/models/dataset:" + dataset + "_model:" + current_model + "_dim:" + str(dim) + "_epoch:" + str(epoch)

        if os.path.exists(current_config):
            print(
                "Configuration already trained. Edit configuration or delete model directory.")
        else:
            results = pipeline(
                training_loop='sLCWA',
                training=train,
                testing=test,
                random_seed=RANDOM_SEED,
                model=current_model,
                model_kwargs=dict(embedding_dim=dim),
                epochs=epoch,
                dimensions=dim,
                device=get_device(),
            )

            results.save_to_directory(current_config)

    return None

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

def load_model(tf, train, valid, test, project_name, dataset):
    # load trained model
    model_number = int(
        input("Enter 1 for TransE\tEnter 2 for BoxE\tEnter 3 RotatE\tEnter 4 DistMult\tEnter 5 CompGCN\t(1-5):"))
    if model_number == 1:
        current_model = "TransE"
    elif model_number == 2:
        current_model = "BoxE"
    elif model_number == 3:
        current_model = "RotatE"
    elif model_number == 4:
        current_model = "DistMult"
    elif model_number == 5:
        current_model = "CompGCN"
    else:
        current_model = None

    dim = int(input("Enter dimension for model %s: " % (current_model)))
    epochs = int(input("Enter number of epochs for model %s: " %
                 (current_model)))
    current_config = "testcases/" + project_name + "/models/dataset:" + \
        dataset + "_model:" + current_model + \
        "_dim:" + str(dim) + "_epoch:" + str(epochs)

    if os.path.exists(current_config):
        model_path = current_config + "/trained_model.pkl"
        trained_model = load(model_path)
        print("model loaded successfully")

        return trained_model, {'selected_model_name': current_model, 'dim': dim, 'epoch': epochs}

    return None

def new_approach(model, k, tf, train, valid, test, queries, tbox, aboxpath):
    
    query_returns= list()
    for i in range(len(queries)):
        atom_returns = list()
        print("Query %i/%i utilizing..." % (i+1, len(queries)))
        for atom in queries[i]["obj"]:
            if atom['obj']['type'] == "concept":
                temp = dict()
                temp['atom'] = atom['obj']
                temp['returned_entities'] = concepts(model, k, tf, train, valid, test, atom['obj'], tbox, aboxpath)
                atom_returns.append(temp)

            elif atom['obj']['type'] == "role":
                temp = dict()
                temp['atom'] = atom['obj']
                temp['returned_entities'] = roles(model, k, tf, train, valid, test, atom['obj'], tbox, aboxpath)
                atom_returns.append(temp)
        query_returns.append(atom_returns)
        print("Query %i/%i finished.\n\n" % (i+1, len(queries)))

    return query_returns

def concepts(dataset, model,k, tf, train, valid, test, query, tbox, aboxpath, candidates = {'variable': None, 'target': None, 'entities': None}):
    entities = dict()
    entities[query.get_var1().get_org_name()] = None
    entities['queried_heads'] = query.get_var1().get_bound()
    entities['var_for_heads'] = query.get_var1().get_org_name()
    X = query_graph_concepts(query, dataset, aboxpath)
    X = pd.DataFrame(data=X)

    # the IDs of the entities
    if not X.empty:
        entityIDs_of_matches = X.iloc[:, 1].values.tolist()
    else:
        print("\nThis Ontology doesnt have any instances of concept %s. However, with rewriting the TBox will handle entailments." % (query.name))
        #press_any_key()
        return entities

    rdf_type_id = tf.relations_to_ids(relations=["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"])[0]
    atom_entity_id = tf.entities_to_ids(entities=["<"+str(query.iri)+">"])[0]

    head_triples = tf.mapped_triples[np.where(np.isin(tf.mapped_triples[:,0], entityIDs_of_matches))]
    head_relations = np.unique(head_triples[:,1], return_index=False)
    tail_triples = tf.mapped_triples[np.where(np.isin(tf.mapped_triples[:,2], entityIDs_of_matches))]
    tail_relations = np.unique(tail_triples[:,1], return_index=False)



    # Predict Heads

    # - Predict top k results from (_, R, X) using PartiallyRestrictedPredictionDataset. Call this set Y
    dataset = predict.PartiallyRestrictedPredictionDataset(
        heads=entityIDs_of_matches, relations=head_relations, target="tail")
    consumer = predict.TopKScoreConsumer(k=k)
    print("\nStep 1/4:\tCurrently predicting for concept %s.\n" % (query.name))
    predict.consume_scores(model, dataset, consumer)
    score_pack = consumer.finalize().process(tf).add_membership_columns(
        training=train, validation=valid, testing=test)
    Y = score_pack.df
    predicted_tails = Y.iloc[:, 4].unique().tolist()

    # - Predict top k results from (Y, R, _) using PartiallyRestrictedPredictionDataset. Call this set Y
    dataset = predict.PartiallyRestrictedPredictionDataset(
        tails=predicted_tails, relations=head_relations, target="head")
    consumer = predict.TopKScoreConsumer(k=k)
    print("\nStep 2/4:\tCurrently predicting for concept %s.\n" % (query.name))
    predict.consume_scores(model, dataset, consumer)
    score_pack = consumer.finalize().process(tf).add_membership_columns(
        training=train, validation=valid, testing=test)
    print(type(score_pack))
    print(score_pack.df.loc)
    Y = score_pack.df
    #Y.to_csv(query["obj"][0]["obj"]["name"] + ".csv")


    # Predict Tails
    dataset = predict.PartiallyRestrictedPredictionDataset(
        tails=entityIDs_of_matches, relations=tail_relations, target="head")
    consumer = predict.TopKScoreConsumer(k=k)
    print("\nStep 3/4:\tCurrently predicting for concept %s.\n" % (query.name))
    predict.consume_scores(model, dataset, consumer)
    score_pack = consumer.finalize().process(tf).add_membership_columns(
        training=train, validation=valid, testing=test)
    YY = score_pack.df
    H = YY.iloc[:, 0].unique().tolist()

    dataset = predict.PartiallyRestrictedPredictionDataset(heads=H, relations=tail_relations, target="tail")
    consumer = predict.TopKScoreConsumer(k=k)
    print("\nStep 4/4:\tCurrently predicting for concept %s.\n" % (query.name))
    predict.consume_scores(model, dataset, consumer)
    score_pack = consumer.finalize().process(tf).add_membership_columns(
        training=train, validation=valid, testing=test)
    YY = score_pack.df


    # create filter
    new_heads = Y.iloc[:, 0].unique().tolist()
    new_tails = YY.iloc[:, 4].unique().tolist()
    new_entities = np.unique(np.concatenate((new_heads, new_tails), 0))
    new_triples = list()

    for entity in new_entities:
        new_triples.append([entity, rdf_type_id, atom_entity_id])

    temp_tensor = torch.tensor(new_triples)
    new_triples = TriplesFactory(
        temp_tensor, entity_to_id=tf.entity_to_id, relation_to_id=tf.relation_to_id)

    print("Filtering predictions for concept %s.\n" % (query.name))
    filter_results = predict.predict_triples(model=model, triples=new_triples).process(factory=tf).add_membership_columns(
        training=train, validation=valid, testing=test).df.sort_values(by=['score'], ascending=False)
    filter_results['still_valid'] = filter_results[["in_training", "in_validation", "in_testing"]].any(axis=1)
    filter_results['score_calibrated'] = filter_results['score'].apply(expit)
    filter_results['score_combined'] = filter_results['score_calibrated']
    entities['head'] = filter_results

    # - Append new results and its score to X

    # return X
    return entities

def roles(model, k, tf, train, valid, test, query, tbox, aboxpath, candidates = {'variable': None, 'target': None, 'entities': None}):
    
    entities = dict()
    entities[query.get_var1().get_org_name()] = None
    entities[query.get_var2().get_org_name()] = None
    entities['queried_heads'] = False
    entities['queried_tails'] = False
    entities['var_for_heads'] = query.get_var1().get_org_name()
    entities['var_for_tails'] = query.get_var2().get_org_name()
    
    # TripleFactory creation for the relevant role.
    try:
        relation = tf.relations_to_ids(relations=["<"+query.iri+">"])[0]
        new_factory = tf.new_with_restriction(relations=[relation])
    except:
        return entities
    
    #If this is a projection step, then we have input candidates for the next iteration
    if not candidates['entities'] is None:
        # if the variable from the previous atom is equal to the head variable in this atom, we predict tails
        if candidates['variable'] == entities['var_for_heads']:
            if candidates['target'] == 'head':
                head_entities = candidates['entities']['head_id'].to_list()
            if candidates['target'] == 'tail':
                head_entities = candidates['entities']['tail_id'].to_list()
            dataset = predict.PartiallyRestrictedPredictionDataset(heads=head_entities, relations=relation, target="tail")

        #We predict the heads with the tails
        elif candidates['variable'] == entities['var_for_tails']:     
            if candidates['target'] == 'head':
                tail_entities = candidates['entities']['head_id'].to_list()
            if candidates['target'] == 'tail':
                tail_entities = candidates['entities']['tail_id'].to_list()
            dataset = predict.PartiallyRestrictedPredictionDataset(tails=tail_entities, relations=relation, target="head")

    #If no candidates, we select them from the ABox. Nececssary for step 1.
    else:
        # Even if both cases can trigger in the code, its never the case that the first atom has two bound variables.
        if query.get_var1().get_bound():
            tail_entities = np.unique(new_factory.mapped_triples[:, 2], return_index=False)
            dataset = predict.PartiallyRestrictedPredictionDataset(tails=tail_entities, relations=relation, target="head")
        if query.get_var2().get_bound():
            head_entities = np.unique(new_factory.mapped_triples[:, 0], return_index=False)
            dataset = predict.PartiallyRestrictedPredictionDataset(heads=head_entities, relations=relation, target="tail")

    consumer = predict.TopKScoreConsumer(k=k)
    print("\nCurrently predicting for role %s.\n" % (query.name))
    predict.consume_scores(model, dataset, consumer)
    score_pack = consumer.finalize().process(tf).add_membership_columns(training=train, validation=valid, testing=test)
    Y = score_pack.df.copy()
    #calibrate with the logistic sigmoid function, expit.
    print(Y)
    Y['still_valid'] = Y[["in_training", "in_validation", "in_testing"]].any(axis=1)
    print(Y)
    Y['score_calibrated'] = Y['score'].apply(expit)
    Y['score_combined'] = Y['score_calibrated']
    
    # Domain, like hasCousin(x, _)
    if query.get_var1().get_bound() and (candidates['variable'] != entities['var_for_heads']):
        
        # Only store the distinct answer entities with the best score. 
        Y_head = Y.drop_duplicates(subset=["head_id"],keep='first', ignore_index=True)
        print(Y_head, Y_head.shape)
        entities['head'] = Y_head
        entities['queried_heads'] = True

    # Range, like hasCousin(_, x)
    if query.get_var2().get_bound() and (candidates['variable'] != entities['var_for_tails']):
        
        # Only store the distinct answer entities with the best score. 
        Y_tail = Y.drop_duplicates(subset=["tail_id"],keep='first', ignore_index=True)
        print(Y_tail, Y_tail.shape)
        entities['tail'] = Y_tail
        entities['queried_tails'] = True

    return entities

def projection(model, k, tf, train, valid, test, query, tbox, aboxpath, candidates = None):
    
    if not candidates is None:

        return None

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

def combine_scores_old(queries, original_query, transductive_models, project_name, dataset):
    list_of_variables = list()
    merged_df = dict()
    for q in queries:
        for atom in q:
            if atom['atom']['type'] == "concept":
                if atom['atom']['var1']['name'] not in list_of_variables:
                    list_of_variables.append(atom['atom']['var1']['name'])
            if atom['atom']['type'] == "role":
                if atom['atom']['var1']['is_bound'] and atom['atom']['var1']['name'] not in list_of_variables:
                    list_of_variables.append(atom['atom']['var1']['name'])
                if atom['atom']['var2']['is_bound'] and atom['atom']['var2']['name'] not in list_of_variables:
                    list_of_variables.append(atom['atom']['var2']['name'])

    #For every unique bound variable
    for bound_variable in list_of_variables:
        merged_df[bound_variable] = pd.DataFrame(data=None, columns=['head_id','head_label','relation_id','relation_label', 'tail_id','tail_label','score','in_training','in_validation','in_testing','from_kge','from_atom','target'])
        
        #Merging dataframes for each variable
        #For every query
        for q in queries:

            #If query doesnt contain any atoms (for some reason)
            if len(q) < 1:
                return None

            #If query contains multiple atoms
            else:
                for atom in q:
                    if atom['atom']['type'] == "concept" and isinstance(atom['returned_entities'],pd.DataFrame):
                        if bound_variable == atom['atom']['var1']['name']:
                            from_kge = [True for i in range(len(atom['returned_entities']))]
                            target = ["head" for i in range(len(atom['returned_entities']))]
                            q_str = [(atom['atom']['name']+"("+atom['atom']['var1']['name']+")") for i in range(len(atom['returned_entities']))]
                            atom['returned_entities']['from_atom'] = q_str
                            atom['returned_entities']['target'] = target
                            atom['returned_entities']['from_kge'] = from_kge
                            merged_df[bound_variable] = merged_df[bound_variable].append(atom['returned_entities'])

                    if atom['atom']['type'] == "role":
                        if isinstance(atom['returned_entities'][bound_variable],pd.DataFrame):
                            from_kge = [True for i in range(len(atom['returned_entities'][bound_variable]))]
                            atom['returned_entities'][bound_variable]['from_kge'] = from_kge
                            if bound_variable == atom['returned_entities']['var_for_heads']:
                                target = ["head" for i in range(len(atom['returned_entities'][bound_variable]))]
                                q_str = [(atom['atom']['name']+"("+atom['atom']['var1']['name']+","+atom['atom']['var2']['name']+")") for i in range(len(atom['returned_entities'][bound_variable]))]
                                atom['returned_entities'][bound_variable]['from_atom'] = q_str
                                atom['returned_entities'][bound_variable]['target'] = target
                                merged_df[bound_variable] = merged_df[bound_variable].append(atom['returned_entities'][bound_variable])

                            if bound_variable == atom['returned_entities']['var_for_tails']:
                                target = ["tail" for i in range(len(atom['returned_entities'][bound_variable]))]
                                q_str = [(atom['atom']['name']+"("+atom['atom']['var1']['name']+","+atom['atom']['var2']['name']+")") for i in range(len(atom['returned_entities'][bound_variable]))]
                                atom['returned_entities'][bound_variable]['from_atom'] = q_str
                                atom['returned_entities'][bound_variable]['target'] = target
                                merged_df[bound_variable] = merged_df[bound_variable].append(atom['returned_entities'][bound_variable])

        #Only get max value. Therefore, in the next step, at the first encounter with a new entity_id, this will indeed be the max score.
        merged_df[bound_variable] = merged_df[bound_variable].sort_values(by=['score'], ascending=False)

    #Combine scores of several queries
    entities_df = dict()
    unique_entities = list()

    #for every variable
    for bound_variable in list_of_variables:
        entities_df[bound_variable] = pd.DataFrame(data=None, columns=['entity_id','entity_label','score','in_training','in_validation','in_testing','from_kge','from_atom'])

        #the heads
        for index, row in merged_df[bound_variable].iterrows():
            if row['target'] == "head":
                if not row['head_label'] in unique_entities:
                    entities_df[bound_variable] = entities_df[bound_variable].append({
                        'entity_id': row['head_id'],
                        'entity_label': row['head_label'],
                        'score': row['score'],
                        'in_training': row['in_training'],
                        'in_validation': row['in_validation'],
                        'in_testing': row['in_testing'],
                        'from_kge': row['from_kge'],
                        'from_atom': row['from_atom']
                    }, ignore_index=True)
                    unique_entities.append(row['head_label'])

            if row['target'] == "tail":
                if not row['tail_label'] in unique_entities:
                    entities_df[bound_variable] = entities_df[bound_variable].append({
                        'entity_id': row['tail_id'],
                        'entity_label': row['tail_label'],
                        'score': row['score'],
                        'in_training': row['in_training'],
                        'in_validation': row['in_validation'],
                        'in_testing': row['in_testing'],
                        'from_kge': row['from_kge'],
                        'from_atom': row['from_atom']
                    }, ignore_index=True)
                    unique_entities.append(row['tail_label'])
        
        print(entities_df[bound_variable])
    press_any_key()

    return entities_df

def update_prediction_pickle(already_predicted_atoms, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(already_predicted_atoms, file, protocol=pickle.HIGHEST_PROTOCOL)

def query_for_different_structures(queries, already_predicted_atoms, dataset, model, model_params, k, tf, train, valid, test, tbox, abox, partial_prediction_path, do_lookup_prestored):  
    projection_types = ['1p', '2p', '3p', 'pi']
    intersection_types = ['2i', '3i']
    projection_and_intersection_types = ['pi', 'ip']
    disjunction_types = ['up','2u']

    print("Current structure: %s\n" % (queries['query_structure']))

    if queries['query_structure'] in projection_types:
        for query in queries['rewritings']:
            already_predicted_variables = list()
            previous_atom = None
            j = 0
            for atom in query.get_body():
                output_candidates = {'variable': None, 'target': None, 'entities': None}
                current_config = {
                    'iri': atom.iri,
                    'type': 'undeclared',
                    'dataset': dataset,
                    'model_name': model_params['selected_model_name'],
                    'model_dim': model_params['dim'],
                    'model_epoch': model_params['epoch'],
                    'prediction_cutoff': k,
                    'domain': False,
                    'range': False,
                    'first': j == 0,
                    'previous': previous_atom,
                    'last': j == len(query.get_body()) - 1 
                    }
                
                #If the atom is a concept
                if isinstance(atom, AtomConcept):
                    current_config['type'] = 'concept'
                    not_predicted = True

                    # Check if the Concept is already predicted
                    i = 0
                    while (not_predicted and i < len(already_predicted_atoms)) and do_lookup_prestored:
                        if current_config == already_predicted_atoms[i]['config']:
                            #If so, return the df
                            atom.set_answer(copy.deepcopy(already_predicted_atoms[i]['result']))
                            already_predicted_variables.append(copy.deepcopy(already_predicted_atoms[i]['result']['variable']))
                            output_candidates['variable'] = copy.deepcopy(already_predicted_atoms[i]['result']['variable'])
                            output_candidates['target'] = copy.deepcopy(already_predicted_atoms[i]['result']['target'])
                            output_candidates['entities'] = copy.deepcopy(already_predicted_atoms[i]['result']['entities'])
                            not_predicted = False
                        i += 1
                    
                    if not_predicted:       
                        # If not, perform prediction on Role
                        
                        #pi
                        if queries['query_structure'] == 'pi':
                            if current_config['first'] or current_config['last']:
                                temp = roles(model, k, tf, train, valid, test, atom, tbox, abox)
                            else:
                                temp = roles(model, k, tf, train, valid, test, atom, tbox, abox, previous_atom.answer)
                        
                        #1p, 2p, 3p
                        else:
                            if not current_config['first']:
                                temp = roles(model, k, tf, train, valid, test, atom, tbox, abox, previous_atom.answer)
                            else:
                                temp = roles(model, k, tf, train, valid, test, atom, tbox, abox)
                        
                        #Projection
                        if temp['queried_heads']:
                            #if not temp['var_for_heads'] in already_predicted_variables:
                            #already_predicted_variables.append(temp['var_for_heads'])
                            output_candidates['variable'] = temp['var_for_heads']
                            output_candidates['target'] = 'head'
                            output_candidates['entities'] = temp[temp['var_for_heads']]

                        # atom.set_answer(temp)
                        # already_predicted_atoms.append({'config': current_config, 'result': temp})
                        # update_prediction_pickle(already_predicted_atoms, partial_prediction_path)

                # If the atom is a a role
                if isinstance(atom, AtomRole):
                    #Current role config
                    current_config['type'] = 'role'            
                    if atom.get_var1().get_bound():
                        current_config['domain'] = True
                    if atom.get_var2().get_bound():
                        current_config['range'] = True

                    not_predicted = True
                    # Check if the Role is already predicted
                    i = 0
                    while (not_predicted and i < len(already_predicted_atoms)) and do_lookup_prestored:
                        if current_config == (already_predicted_atoms[i]['config']):
                            #If so, return the df
                            atom.set_answer(copy.deepcopy(already_predicted_atoms[i]['result']))
                            already_predicted_variables.append(copy.deepcopy(already_predicted_atoms[i]['result']['variable']))
                            output_candidates['variable'] = copy.deepcopy(already_predicted_atoms[i]['result']['variable'])
                            output_candidates['target'] = copy.deepcopy(already_predicted_atoms[i]['result']['target'])
                            output_candidates['entities'] = copy.deepcopy(already_predicted_atoms[i]['result']['entities'])
                            not_predicted = False
                        i += 1

                    if not_predicted:       
                        # If not, perform prediction on Role
                        
                        #pi
                        if queries['query_structure'] == 'pi':
                            if current_config['first']:
                                temp = roles(model, k, tf, train, valid, test, atom, tbox, abox)
                            else:
                                temp = roles(model, k, tf, train, valid, test, atom, tbox, abox, previous_atom.answer)
                        
                        #1p, 2p, 3p
                        else:
                            if not current_config['first']:
                                temp = roles(model, k, tf, train, valid, test, atom, tbox, abox, previous_atom.answer)
                            else:
                                temp = roles(model, k, tf, train, valid, test, atom, tbox, abox)
                        
                        #Projection
                        if temp['queried_heads']:
                            #if not temp['var_for_heads'] in already_predicted_variables:
                                #already_predicted_variables.append(temp['var_for_heads'])
                            output_candidates['variable'] = temp['var_for_heads']
                            output_candidates['target'] = 'head'
                            output_candidates['entities'] = temp['head']


                        if temp['queried_tails']:
                            #if not temp['var_for_tails'] in already_predicted_variables:
                                #already_predicted_variables.append(temp['var_for_tails'])
                            output_candidates['variable'] = temp['var_for_tails']
                            output_candidates['target'] = 'tail'
                            output_candidates['entities'] = temp['tail']
                        
                atom.set_answer(output_candidates)


                # Combine scores
                if (not current_config['first']) and previous_atom.answer is not None and atom.answer is not None:
                    
                    output_candidates['entities'] = combine_scores(previous_atom, atom)
                    atom.set_answer(output_candidates)

                already_predicted_atoms.append({'config': current_config, 'result': output_candidates})
                update_prediction_pickle(already_predicted_atoms, partial_prediction_path)


                previous_atom = copy.deepcopy(atom)
                
                # j is used for keeping track of the atom number in the query
                j += 1

                # Set the final query answer in the Query-body
                if j == len(query.get_body()):
                    column_labels = ['entity_id', 'entity_label', 'still_valid', 'score_combined']
                    entities = pd.DataFrame(data=None, columns=column_labels)

                    if query.answer['entities'] is not None:
                        df_answer = q.answer['entities']

                        if query.answer['target'] == 'head':
                            df_answer = df_answer[['head_id', 'head_label', 'still_valid', 'score_combined']]
                            df_answer = df_answer.rename(columns={'head_id': 'entity_id', 'head_label': 'entity_label'})
                        elif query.answer['target'] == 'tail':
                            df_answer = df_answer[['tail_id', 'tail_label', 'still_valid', 'score_combined']]
                            df_answer = df_answer.rename(columns={'tail_id': 'entity_id', 'tail_label': 'entity_label'})
                            #Let entities be the total dataframe of all entities.
                        entities = pd.concat([entities, df_answer], axis=0, ignore_index=True).sort_values(['score_combined'], ignore_index=True, ascending=False).drop_duplicates(subset=['entity_id'], keep='first', ignore_index=True)
                    print("Final:" + entities)
                    query.set_answer(entities)
      
    if queries['query_structure'] in intersection_types:
        for query in queries['rewritings']:
            already_predicted_variables = list()
            previous_atom = None
            j = 0
            for atom in query.get_body():
                output_candidates = {'variable': None, 'target': None, 'entities': None}
                current_config = {
                    'iri': atom.iri,
                    'type': 'undeclared',
                    'dataset': dataset,
                    'model_name': model_params['selected_model_name'],
                    'model_dim': model_params['dim'],
                    'model_epoch': model_params['epoch'],
                    'prediction_cutoff': k,
                    'domain': False,
                    'range': False,
                    'first': j == 0,
                    'previous': previous_atom,
                    'last': j == len(query.get_body()) - 1 
                    }
                
                #If the atom is a concept
                if isinstance(atom, AtomConcept):
                    current_config['type'] = 'concept'
                    not_predicted = True

                    # Check if the Concept is already predicted
                    i = 0
                    while (not_predicted and i < len(already_predicted_atoms)) and do_lookup_prestored:
                        if current_config == already_predicted_atoms[i]['config']:
                            #If so, return the df
                            atom.set_answer(copy.deepcopy(already_predicted_atoms[i]['result']))
                            already_predicted_variables.append(copy.deepcopy(already_predicted_atoms[i]['result']['variable']))
                            output_candidates['variable'] = copy.deepcopy(already_predicted_atoms[i]['result']['variable'])
                            output_candidates['target'] = copy.deepcopy(already_predicted_atoms[i]['result']['target'])
                            output_candidates['entities'] = copy.deepcopy(already_predicted_atoms[i]['result']['entities'])
                            not_predicted = False
                        i += 1
                    
                    if not_predicted:       
                        # If not, perform prediction on Role

                        temp = concepts(dataset, model, k, tf, train, valid, test, atom, tbox, abox)
                        
                        #Projection
                        if temp['queried_heads']:
                            #if not temp['var_for_heads'] in already_predicted_variables:
                            #already_predicted_variables.append(temp['var_for_heads'])
                            output_candidates['variable'] = temp['var_for_heads']
                            output_candidates['target'] = 'head'
                            output_candidates['entities'] = temp[temp['var_for_heads']]

                        # atom.set_answer(temp)
                        # already_predicted_atoms.append({'config': current_config, 'result': temp})
                        # update_prediction_pickle(already_predicted_atoms, partial_prediction_path)

                # If the atom is a a role
                if isinstance(atom, AtomRole):
                    #Current role config
                    current_config['type'] = 'role'            
                    if atom.get_var1().get_bound():
                        current_config['domain'] = True
                    if atom.get_var2().get_bound():
                        current_config['range'] = True

                    not_predicted = True
                    # Check if the Role is already predicted
                    i = 0
                    while (not_predicted and i < len(already_predicted_atoms)) and do_lookup_prestored:
                        if current_config == (already_predicted_atoms[i]['config']):
                            #If so, return the df
                            atom.set_answer(copy.deepcopy(already_predicted_atoms[i]['result']))
                            already_predicted_variables.append(copy.deepcopy(already_predicted_atoms[i]['result']['variable']))
                            output_candidates['variable'] = copy.deepcopy(already_predicted_atoms[i]['result']['variable'])
                            output_candidates['target'] = copy.deepcopy(already_predicted_atoms[i]['result']['target'])
                            output_candidates['entities'] = copy.deepcopy(already_predicted_atoms[i]['result']['entities'])
                            not_predicted = False
                        i += 1

                    if not_predicted:       
                        # If not, perform prediction on Role

                        temp = roles(model, k, tf, train, valid, test, atom, tbox, abox)
                        
                        #Projection
                        if temp['queried_heads']:
                            output_candidates['variable'] = temp['var_for_heads']
                            output_candidates['target'] = 'head'
                            output_candidates['entities'] = temp['head']

                        if temp['queried_tails']:
                            output_candidates['variable'] = temp['var_for_tails']
                            output_candidates['target'] = 'tail'
                            output_candidates['entities'] = temp['tail']
                        
                atom.set_answer(output_candidates)
                
                # j is used for keeping track of the atom number in the query
                j += 1

            #standardize answer DataFrame
            list_of_answers = list()
            for q in query.get_body():
                if q.answer['entities'] is not None:
                    df_answer = q.answer['entities']

                    if q.answer['target'] == 'head':
                        df_answer = df_answer[['head_id', 'head_label', 'still_valid', 'score_combined']]
                        df_answer = df_answer.rename(columns={'head_id': 'entity_id', 'head_label': 'entity_label'})
                    elif q.answer['target'] == 'tail':
                        df_answer = df_answer[['tail_id', 'tail_label', 'still_valid', 'score_combined']]
                        df_answer = df_answer.rename(columns={'tail_id': 'entity_id', 'tail_label': 'entity_label'})
                        #Let entities be the total dataframe of all entities.
                q.answer['entities'] = df_answer
                list_of_answers.append(df_answer)
            
            #Intersect results
            if queries['query_structure'] == '2i':
                entities = list_of_answers[0][ list_of_answers[0]['entity_id'].isin(list_of_answers[1]['entity_id']) ]
            if queries['query_structure'] == '3i':
                entities = list_of_answers[0][ list_of_answers[0]['entity_id'].isin(list_of_answers[1]['entity_id']) ]
                entities = entities[entities['entity_id'].isin(list_of_answers[2]['entity_id']) ]
            
            entities = entities.sort_values(['score_combined'], ignore_index=True, ascending=False).drop_duplicates(subset=['entity_id'], ignore_index=True, keep='first')
            
            print(entities)
            query.set_answer(entities)

    if queries['query_structure'] == 'pi':
        print("debug")
        try:
            temp = roles(model, k, tf, train, valid, test, atom, tbox, abox, previous_atom.answer)
        except:
            temp = roles(model, k, tf, train, valid, test, atom, tbox, abox)
        temp = concepts(dataset, model, k, tf, train, valid, test, atom, tbox, abox)


        return None
    
    if queries['query_structure'] == 'ip':
        return None

    if queries['query_structure'] in disjunction_types:
        return queries
    
    queries['answer'] = combine_entities(queries)

    return queries

def combine_entities(predicted_queries):
    column_labels = ['entity_id', 'entity_label', 'still_valid', 'score_combined']
    entities = pd.DataFrame(data=None, columns=column_labels)

    for q in predicted_queries['rewritings']:
        if q.answer.answer['entities'] is not None:
            df_answer = q.answer.answer['entities']

            # if q.answer.answer['target'] == 'head':
            #     df_answer = df_answer[['head_id', 'head_label', 'still_valid', 'score_combined']]
            #     df_answer = df_answer.rename(columns={'head_id': 'entity_id', 'head_label': 'entity_label'})
            # elif q.answer.answer['target'] == 'tail':
            #     df_answer = df_answer[['tail_id', 'tail_label', 'still_valid', 'score_combined']]
            #     df_answer = df_answer.rename(columns={'tail_id': 'entity_id', 'tail_label': 'entity_label'})

            #Let entities be the total dataframe of all entities.
            entities = pd.concat([entities, df_answer], axis=0, ignore_index=True).sort_values(['score_combined'], ignore_index=True, ascending=False).drop_duplicates(subset=['entity_id'], keep='first', ignore_index=True)
    print(entities)
    return entities

def combine_scores(previous_atom, current_atom):
    current_answers = current_atom.answer['entities']
    previous_answers = previous_atom.answer['entities']

    if isinstance(previous_atom, AtomRole) and isinstance(current_atom, AtomRole):
        #if previous prediction had the target: tail.
        if previous_atom.get_var2().get_bound(): 

            # if the target variable from previous is equal to variable 1 in current atom
            
            #r1(x,y), r2(y,z)
            if previous_atom.get_var2().get_org_name() == previous_atom.get_answer()['variable'] and previous_atom.get_answer()['variable'] == current_atom.get_var1().get_org_name():
                
                # In previous prediction, select the row with the highest score, and apply t-norm to the scores and save it to combined_score
                current_answers['score_combined'] = current_answers.apply(lambda x: t_norm(x['score_combined'], previous_answers.set_index('tail_id')['score_combined'].get(x['head_id'], 1)), axis=1)
            
            # if this target variable is equal to variable 2 in current atom
            #r1(x,y), r2(z,y)
            elif previous_atom.get_var2().get_org_name() == previous_atom.get_answer()['variable'] and previous_atom.get_answer()['variable'] == current_atom.get_var2().get_org_name():

                # in recent prediction, select input top target entity
                current_answers['score_combined'] = current_answers.apply(lambda x: t_norm(x['score_combined'], previous_answers.set_index('tail_id')['score_combined'].get(x['tail_id'], 1)), axis=1)

        #if previous prediction yielded target head
        if previous_atom.get_var1().get_bound(): 

            # if the target variable from previous is equal to variable 1 in current atom
            # r1(y,x),r2(y,z)
            if previous_atom.get_var1().get_org_name() == previous_atom.get_answer()['variable'] and previous_atom.get_answer()['variable'] == current_atom.get_var1().get_org_name():
                
                # In previous prediction, select the row with the highest score, and apply t-norm to the scores and save it to combined_score
                current_answers['score_combined'] = current_answers.apply(lambda x: t_norm(x['score_combined'], previous_answers.set_index('head_id')['score_combined'].get(x['head_id'], 1)), axis=1)
                    
            # if this target variable is equal to variable 2 in current atom
            # # r1(y,x),r2(z,y)
            elif previous_atom.get_var1().get_org_name() == previous_atom.get_answer()['variable'] and previous_atom.get_answer()['variable'] == current_atom.get_var2().get_org_name():
                
                # in recent prediction, select input top target entity
                current_answers['score_combined'] = current_answers.apply(lambda x: t_norm(x['score_combined'], previous_answers.set_index('head_id')['score_combined'].get(x['tail_id'], 1)), axis=1)
        

    elif isinstance(previous_atom, AtomRole) and isinstance(current_atom, AtomConcept):
        #if previous prediction had the target: tail.
        if previous_atom.get_var2().get_bound(): 

            # if the target variable from previous is equal to variable 1 in current atom
            
            #r1(x,y), c(y)
            if previous_atom.get_var2().get_org_name() == previous_atom.get_answer()['variable'] and previous_atom.get_answer()['variable'] == current_atom.get_var1().get_org_name():
                
                # In previous prediction, select the row with the highest score, and apply t-norm to the scores and save it to combined_score
                current_answers['score_combined'] = current_answers.apply(lambda x: t_norm(x['score_combined'], previous_answers.set_index('tail_id')['score_combined'].get(x['head_id'], 1)), axis=1)

        #if previous prediction yielded target head
        if previous_atom.get_var1().get_bound(): 

            # if the target variable from previous is equal to variable 1 in current atom
            # r1(y,x),c(y)
            if previous_atom.get_var1().get_org_name() == previous_atom.get_answer()['variable'] and previous_atom.get_answer()['variable'] == current_atom.get_var1().get_org_name():
                
                # In previous prediction, select the row with the highest score, and apply t-norm to the scores and save it to combined_score
                current_answers['score_combined'] = current_answers.apply(lambda x: t_norm(x['score_combined'], previous_answers.set_index('head_id')['score_combined'].get(x['head_id'], 1)), axis=1)
                    
    elif isinstance(previous_atom, AtomConcept) and isinstance(current_atom, AtomRole):
        #if previous prediction yielded target head
        if previous_atom.get_var1().get_bound(): 

            # if the target variable from previous is equal to variable 1 in current atom
            # c(x),r2(x,y)
            if previous_atom.get_var1().get_org_name() == previous_atom.get_answer()['variable'] and previous_atom.get_answer()['variable'] == current_atom.get_var1().get_org_name():
                
                # In previous prediction, select the row with the highest score, and apply t-norm to the scores and save it to combined_score
                current_answers['score_combined'] = current_answers.apply(lambda x: t_norm(x['score_combined'], previous_answers.set_index('head_id')['score_combined'].get(x['head_id'], 1)), axis=1)
                    
            # if this target variable is equal to variable 2 in current atom
            # # c(x),r2(y,x)
            elif previous_atom.get_var1().get_org_name() == previous_atom.get_answer()['variable'] and previous_atom.get_answer()['variable'] == current_atom.get_var2().get_org_name():
                
                # in recent prediction, select input top target entity
                current_answers['score_combined'] = current_answers.apply(lambda x: t_norm(x['score_combined'], previous_answers.set_index('head_id')['score_combined'].get(x['tail_id'], 1)), axis=1)

    elif isinstance(previous_atom, AtomConcept) and isinstance(current_atom, AtomConcept):
        #if previous prediction yielded target head
        if previous_atom.get_var1().get_bound(): 
           
           # c1(x),c2(x)
           if previous_atom.get_var1().get_org_name() == previous_atom.get_answer()['variable'] and previous_atom.get_answer()['variable'] == current_atom.get_var1().get_org_name():
                
                # In previous prediction, select the row with the highest score, and apply t-norm to the scores and save it to combined_score
                current_answers['score_combined'] = current_answers.apply(lambda x: t_norm(x['score_combined'], previous_answers.set_index('head_id')['score_combined'].get(x['head_id'], 1)), axis=1)

    current_answers = current_answers.sort_values(by='score_combined', ascending=False, ignore_index=True)
    print(current_answers)

    
    return current_answers

def t_norm(a, b):
    return a*b

def tco_norm(a, b):
    return max(a, b)

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

def main():
    # SETTINGS
    tbox_import_file = "dbpedia15k.owl"
    tbox_ontology = None
    query = None
    dataset = "dbpedia15k"
    t_box_path = "dataset/"+dataset+"/tbox/"+tbox_import_file
    a_box_path = "dataset/"+dataset+"/abox/transductive/"
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

    partial_results_path  = "testcases/" + project_name + "/already_predicted_atoms.pickle"
    
    if os.path.exists(partial_results_path):
        with open(partial_results_path, 'rb') as file:
            already_predicted_atoms = pickle.load(file)
    else:
        already_predicted_atoms = list()

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
                query_generator.main(project_name, number_of_queries_per_structure)
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
                    parsed_generated_queries[key].append(query_parser.query_structure_parsing(q, key, tbox_ontology))
            
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
                received_entities_kge_queries = new_approach(current_model,100, tf, train, valid, test, [query_object], tbox_ontology, a_box_path)
                entities_df = combine_scores_old(received_entities_kge_queries, query, transductive_models, project_name, dataset)      
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
                received_entities_kge_queries = new_approach(current_model,100, tf, train, valid, test, parsed_entailed_queries_subset, tbox_ontology, a_box_path)
                entities_df = combine_scores_old(received_entities_kge_queries, query, transductive_models, project_name, dataset)      
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
                received_entities_kge_queries = new_approach(current_model,100, tf, train, valid, test, parsed_entailed_queries, tbox_ontology, a_box_path)
                entities_df = combine_scores_old(received_entities_kge_queries, query, transductive_models, project_name, dataset)      
                write_results_to_file_kge(entities_df, query, project_name)

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

                #Predict
                #for each query structure
                for query_structure in parsed_generated_queries.keys():

                    #for each query in that structure
                    for query_dict in parsed_generated_queries[query_structure]:
                        query_dict = query_for_different_structures(query_dict,already_predicted_atoms,dataset,current_model, current_model_params, 100, tf, train, valid, test, tbox_ontology, a_box_path, partial_results_path, use_storage_file)

                print("debug_stop")
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
