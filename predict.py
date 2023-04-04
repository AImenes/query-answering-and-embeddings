import pandas as pd
import numpy as np
import pickle
import torch

from pykeen import predict
from pykeen.triples import TriplesFactory
from scipy.special import expit

from kglookup import *
from utilities import *

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

#Intersect dataframe df1 and df2 and return the final dataframe with the top scores
def intersection(df1, df2):

    #create a new Dataframe where it keeps the common entity_ids from both dataframes
    entities = df1[df1['entity_id'].isin(df2['entity_id']) ]

    # sort the new dataframe on scores. If duplicates, store the best score. That is, the first occuring instance of an entity_id because of the previous sort.
    entities = entities.sort_values(['score_combined'], ignore_index=True, ascending=False).drop_duplicates(subset=['entity_id'], ignore_index=True, keep='first')

    return entities

# Take the union of dataframe df1 and df2 and return the final dataframe with the top scores
def disjunction(df1, df2):

    # Join the rows from both dataframes.
    entities = pd.concat([df1, df2], axis=0, ignore_index=True)

    # sort the new dataframe on scores. If duplicates, store the best score. That is, the first occuring instance of an entity_id because of the previous sort.
    entities = entities.sort_values(['score_combined'], ignore_index=True, ascending=False).drop_duplicates(subset=['entity_id'], keep='first', ignore_index=True)

    return entities

# Project the previous
def projection():
    return None

#Perform prediction
def prediction():
    
    #if concept
        #predict concept
    #if role
        #predict role

    return None

# Handle the base-prediction cases
def base_predictions(query, bases, current_config):
    
    #for each atom in query
    for g in query:

        # for roles
        if isinstance(g, AtomRole):

            # if it has an unbound, and target is tail
            if g.var1.unbound and g.var2.bound:

                # predict
                result = pd.Dataframe()

                # store in bases
            
            # if it has an unbound, and target is head
            if g.var2.unbound and g.var1.bound:

                # predict
                break
                # store in bases

        elif isinstance(g, AtomConcept):

            #if the concept is bound (which should always be the case)
            if g.var1.bound:
                break

    return bases

# Remove head_id, tail_ids and subsitite with entity_id
def standardize_dataframe():
    return None

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

def extract_boundness_of_query_variables(query):
    boundness = dict()

    for atom in query:
        if not atom.var1.original_entry_name in boundness:
            boundness[atom.var1.original_entry_name] = {
                'unbound': atom.var1.unbound,
                'shared': atom.var1.bound and not atom.var1.distinguished,   
                'distinguished': atom.var1.distinguished
            }
        
        if isinstance(query, AtomRole):
            if not atom.var2.original_entry_name in boundness:
                boundness[atom.var2.original_entry_name] = {
                    'unbound': atom.var2.unbound,
                    'shared': atom.var2.bound and not atom.var2.distinguished,
                    'distinguished': atom.var2.distinguished
                }

    return boundness

    

def query_pipeline(query):

    #identify the distinguished variable
    distinguished = '?w'
    
    ## Performing base-case predictions ##
    #   if one of the variables is unbound, then we know its a 'root leaf', and can safely be predicted initially.
    #   We call these atoms base-predictions, and these can be stored for later lookups as long as we save the parameters used for the prediction.
    #   This we do with the current-config dictionary.
  
    someting = base_predictions

    #   We write to file for a latter execution of this code.
    # write_to_some_file(someting)
    
    # Identify the variables in the current query, and create a boundness-dictionary which keeps track of the 
    # entities bound to a variable
    boundness = extract_boundness_of_query_variables

    # Bind base-predictions to variables

    # If the variable already contains entries, perform intersection


    return None

def predict_parsed_queries(all_queries,base_cases,dataset,current_model, current_model_params, k, tf, train, valid, test, tbox_ontology, a_box_path, partial_results_path, use_storage_file):

    # For each original structure type
    for query_structure in all_queries.keys():

                    #for each query in that structure
                    for query in all_queries[query_structure]:
                        query['answer'] = 

    
    return all_queries

