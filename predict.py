import pandas as pd
import numpy as np
import torch

from pykeen import predict
from pykeen.triples import TriplesFactory
from scipy.special import expit

from kglookup import *
from utilities import *


"""

The prediction of queries

In this file, we find the methods utilized for prediction.

We define a term 'base case' as:
 - For concepts:
    - All concepts that have a bound variable (Which should be all concepts. Otherwise there is something wrong.)
 - For roles:
    - All roles where exactly one of the variables is unbound. This means that the atom is not used in a projection step.

We start of by identifying all the base-cases used.


"""

#Predict base case for concepts
def concepts(dataset, model,k, tf, train, valid, test, atom, aboxpath):
    
    # get candidates if rdf_type exists in the ABox, otherwise return None.
    X = query_graph_concepts(atom, dataset, aboxpath)
    X = pd.DataFrame(data=X)

    # the IDs of the entities
    if not X.empty:
        entityIDs_of_matches = X.iloc[:, 1].values.tolist()
    else:
        print("\nThis Ontology doesnt have any instances of concept %s. However, with rewriting the TBox will handle entailments." % (atom.name))
        
        #Return empty Dataframe on same format as base-cases dataframes.
        cols = ['head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label', 'score', 'in_training', 'in_validation', 'in_testing']
        return pd.DataFrame(columns=cols)

    rdf_type_id = tf.relations_to_ids(relations=["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"])[0]
    atom_entity_id = tf.entities_to_ids(entities=["<"+str(atom.iri)+">"])[0]

    head_triples = tf.mapped_triples[np.where(np.isin(tf.mapped_triples[:,0], entityIDs_of_matches))]
    head_relations = np.unique(head_triples[:,1], return_index=False)
    tail_triples = tf.mapped_triples[np.where(np.isin(tf.mapped_triples[:,2], entityIDs_of_matches))]
    tail_relations = np.unique(tail_triples[:,1], return_index=False)



    # Predict Heads

    # - Predict top k results from (_, R, X) using PartiallyRestrictedPredictionDataset. Call this set Y
    dataset = predict.PartiallyRestrictedPredictionDataset(
        heads=entityIDs_of_matches, relations=head_relations, target="tail")
    consumer = predict.TopKScoreConsumer(k=k)
    print("\nPredicting for concept %s, where k = %i. Step 1/5\n" % (atom.name, k))
    predict.consume_scores(model, dataset, consumer)
    score_pack = consumer.finalize().process(tf).add_membership_columns(
        training=train, validation=valid, testing=test)
    Y = score_pack.df
    predicted_tails = Y.iloc[:, 4].unique().tolist()

    # - Predict top k results from (Y, R, _) using PartiallyRestrictedPredictionDataset. Call this set Y
    dataset = predict.PartiallyRestrictedPredictionDataset(
        tails=predicted_tails, relations=head_relations, target="head")
    consumer = predict.TopKScoreConsumer(k=k)
    print("\nPredicting for concept %s, where k = %i. Step 2/5\n" % (atom.name, k))
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
    print("\nPredicting for concept %s, where k = %i. Step 3/5\n" % (atom.name, k))
    predict.consume_scores(model, dataset, consumer)
    score_pack = consumer.finalize().process(tf).add_membership_columns(
        training=train, validation=valid, testing=test)
    YY = score_pack.df
    H = YY.iloc[:, 0].unique().tolist()

    dataset = predict.PartiallyRestrictedPredictionDataset(heads=H, relations=tail_relations, target="tail")
    consumer = predict.TopKScoreConsumer(k=k)
    print("\nPredicting for concept %s, where k = %i. Step 4/5\n" % (atom.name, k))
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

    print("\Filtering predictions for concept %s, where k = %i. Step 5/5\n" % (atom.name, k))
    filter_results = predict.predict_triples(model=model, triples=new_triples).process(factory=tf).add_membership_columns(
        training=train, validation=valid, testing=test).df.sort_values(by=['score'], ascending=False)
    #filter_results['still_valid'] = filter_results[["in_training", "in_validation", "in_testing"]].any(axis=1)
    #filter_results['score_calibrated'] = filter_results['score'].apply(expit)
    #filter_results['score_combined'] = filter_results['score_calibrated']
    #entities['head'] = filter_results

    # - Append new results and its score to X

    # return X
    return filter_results

#Predict base case for roles
def roles(model, k, tf, train, valid, test, atom, target):     
    # TripleFactory creation for the relevant role. 
    relation = tf.relations_to_ids(relations=["<"+atom.iri+">"])[0]
    new_factory = tf.new_with_restriction(relations=[relation])

    #Creating the dataset with respect to the target
    if target == 'head':
        tail_entities = np.unique(new_factory.mapped_triples[:, 2], return_index=False)
        dataset = predict.PartiallyRestrictedPredictionDataset(tails=tail_entities, relations=relation, target="head")
    if target == 'tail':
        head_entities = np.unique(new_factory.mapped_triples[:, 0], return_index=False)
        dataset = predict.PartiallyRestrictedPredictionDataset(heads=head_entities, relations=relation, target="tail")

    # Perform prediction using PyKEEN Predict.
    consumer = predict.TopKScoreConsumer(k=k)
    print("\nPrediction base case for role %s, where k = %i.\n" % (atom.name, k))
    predict.consume_scores(model, dataset, consumer)
    score_pack = consumer.finalize().process(tf).add_membership_columns(training=train, validation=valid, testing=test)
    role_df = score_pack.df

    # We want the score between 0 and 1. We calibrate it using logistic sigmoid function, and add it as a column
    role_df['score_calibrated'] = role_df['score'].apply(expit)

    # We add a column to check if at least one of the three subsets are true. This will be used to verify in the end if each atom prediction in a query has returned a true prediction along the way.
    role_df['still_valid'] = role_df[["in_training", "in_validation", "in_testing"]].any(axis=1)

    # Return the dataframe role_df
    return role_df

# Project the previous
def projection():
    return None

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



#Perform prediction
def prediction():
    
    #if concept
        #predict concept
    #if role
        #predict role

    return None

def new_base_case(base_cases, config):
    for b in base_cases:
        if b['config'] == config:
            return False
    return True

# Handle the base-prediction cases
def base_predictions(query, base_cases, base_case_path, dataset, model, model_params, k, tf, train, valid, test, aboxpath):
    
    #for each atom in query
    for g in query.body:

        #CONCEPTS
        if isinstance(g, AtomConcept):

            #if the concept is bound (which should always be the case)
            if g.var1.bound:

                current_config = {'dataset': dataset, 'model': model_params, 'k': k, 'iri': g.iri, 'target': 'head'}

                if new_base_case(base_cases, current_config):
                    predicted_entities_df = concepts(dataset, model, k, tf, train, valid, test, g, aboxpath)
                

        #ROLES
        if isinstance(g, AtomRole):

            # if it has an unbound, and target is tail
            if g.var1.unbound and g.var2.bound:
                
                current_config = {'dataset': dataset, 'model': model_params, 'k': k, 'iri': g.iri, 'target': 'tail'}
                
                if new_base_case(base_cases, current_config):
                    predicted_entities_df = roles(model, k, tf, train, valid, test, g, current_config['target'])
                
            
            # if it has an unbound, and target is head
            if g.var2.unbound and g.var1.bound:

                current_config = {'dataset': dataset, 'model': model_params, 'k': k, 'iri': g.iri, 'target': 'head'}

                if new_base_case(base_cases, current_config):
                    predicted_entities_df = roles(model, k, tf, train, valid, test, g, current_config['target'])

        
        #add the prediction to the list of dictionaries, base_cases.
        if new_base_case(base_cases, current_config):
            base_cases.append({'config': current_config, 'entities_df': predicted_entities_df})
            update_prediction_pickle(base_cases, base_case_path)

    return base_cases

# Remove head_id, tail_ids and subsitite with entity_id
def standardize_dataframe():
    #Add column of expit

    #remove and rename columns

    #return the dataframe
    
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

    for atom in query.body:
        if not atom.var1.original_entry_name in boundness:
            boundness[atom.var1.original_entry_name] = {
                'unbound': atom.var1.unbound,
                'shared': atom.var1.bound and not atom.var1.distinguished,   
                'distinguished': atom.var1.distinguished
            }
        
        if isinstance(atom, AtomRole):
            if not atom.var2.original_entry_name in boundness:
                boundness[atom.var2.original_entry_name] = {
                    'unbound': atom.var2.unbound,
                    'shared': atom.var2.bound and not atom.var2.distinguished,
                    'distinguished': atom.var2.distinguished
                }

    return boundness

def query_pipeline(query, base_cases, base_cases_path, dataset, model, model_params, k, tf, train, valid, test, aboxpath):

    #identify the distinguished variable
    distinguished_variable = query['q1'].head.var1.original_entry_name
    
    ## Performing base-case predictions ##
    #   if one of the variables is unbound, then we know its a 'root leaf', and can safely be predicted initially.
    #   We call these atoms base-predictions, and these can be stored for later lookups as long as we save the parameters used for the prediction.
    #   This we do with the current-config dictionary.
    for querybody in query['rewritings']:
            base_cases = base_predictions(querybody, base_cases, base_cases_path, dataset, model, model_params, k, tf, train, valid, test, aboxpath)

    # For each query formulation, identify the variables in the current query, and create a boundness-dictionary which keeps track of the 
    # entities bound to a variable
    for querybody in query['rewritings']:
        entities = dict()
        boundness = extract_boundness_of_query_variables(querybody)
        
        # Bind base-predictions to variables and standardize the dataframe
        for atom in querybody.body:
            atom.var 

    # If the variable already contains entries, perform intersection

    return entities

def predict_parsed_queries(all_queries,base_cases, base_cases_path, dataset,model, model_params, k, tf, train, valid, test, tbox_ontology, aboxpath, use_storage_file):

    # For each original structure type
    for query_structure in all_queries.keys():

                    #for each query in that structure
                    for query in all_queries[query_structure]:
                        query['answer'] = query_pipeline(query, base_cases, base_cases_path, dataset, model, model_params, k, tf, train, valid, test, aboxpath)

    
    return all_queries

