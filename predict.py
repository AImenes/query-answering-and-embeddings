import pandas as pd
import numpy as np
import torch
import json

from pykeen import predict
from pykeen.triples import TriplesFactory
from scipy.special import expit
from copy import deepcopy
from functools import reduce

from kglookup import *
from utilities import *
from metrices import * 


def t_norm(a, b):
    """
    The T-norm between two scoring values.
    
    Args:
        a (float): Scoring float
        b (float): Scoring float 
    Returns:
        float: The t-norm between them. In this case multiplication.
    """
    return a*b

def tco_norm(a, b):
    """
    The T-conorm between two scoring values.
    
    Args:
        a (float): Scoring float
        b (float): Scoring float 
    Returns:
        float: The t-norm between them. In this case max.
    """
    return max(a, b)

def and_gate(statement1, statement2):
    """
    Perform the logical AND operation between two boolean statements.

    Args:
        statement1 (bool): The first boolean statement.
        statement2 (bool): The second boolean statement.

    Returns:
        bool: The result of the logical AND operation between statement1 and statement2.
    """

    # Return the result of the logical AND operation between the two input statements
    return statement1 and statement2

def concepts(dataset, model, k, tf, train, valid, test, atom, aboxpath):
    """
    Given a dataset, model, and parameters, this function predicts the top k triples for a given concept
    and filters the results based on the rdf:type. It returns a DataFrame containing the predicted triples
    and additional information. If there does not exist candidates for the concept in the KG, it will return
    empty.

    Concept pipeline:
        1.  Assume we have a list of triples, X, where the concept Father
            is the object according only to lookups in the training data

        2.  Identify all relations X is used with in the ABox, call this set R.

        3.  For tails, 
            3.1     We can query the KGE for entities y, such that (y, r, x)
                    with x is in X, r in R, and keep top k, call this set Y.

            3.2     Then, we can query the KGE for entities z such that (y, r, z)
                    yield a top k ranking of triples with r in R and y in Y. Let us call this
                    set Z.
            
        4    For heads, 
            4.1     We can query the KGE for entities y, such that (y, r, x)
                    with x is in X, r in R, and keep top k, call this set Y.

            4.2     Then, we can query the KGE for entities z such that (y, r, z)
                    yield a top k ranking of triples with r in R and y in Y. Let us call this
                    set Z.
        
        5.  We do a final scoring prediction on these triples, sort them by score and drop duplicates (that is, dups with lower score)

    Args:
        dataset: The input dataset.
        model: The knowledge graph embedding model.
        k (int): The number of top predictions to consider.
        tf (TriplesFactory): A TriplesFactory instance containing the original triples.
        train: Training set of the knowledge graph.
        valid: Validation set of the knowledge graph.
        test: Test set of the knowledge graph.
        atom: The input concept for which the predictions are made.
        aboxpath (str): The path to the ABox file.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered predicted triples and additional information
                      such as 'score', 'in_training', 'in_validation', 'in_testing', 'still_valid',
                      'score_calibrated', and 'score_combined'.
    """
    
    # get candidates if rdf_type exists in the ABox, otherwise return None.
    X = query_graph_concepts(atom, dataset, aboxpath)
    X = pd.DataFrame(data=X)

    # the IDs of the entities
    if not X.empty:
        entityIDs_of_matches = X.iloc[:, 1].values.tolist()
    
    # If they do not exist, return an empty dataframe with the same columns as if a prediction was successful.
    else:
        print("\nThis Ontology doesnt have any instances of concept %s. However, with rewriting the TBox will handle entailments." % (atom.name))
        
        #Return empty Dataframe on same format as base-cases dataframes.
        cols = ['head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label', 'score', 'in_training', 'in_validation', 'in_testing', 'still_valid', 'origin', 'score_calibrated', 'score_combined']
        return pd.DataFrame(columns=cols)

    # Find the ID for rdf:type and the current atom.
    rdf_type_id = tf.relations_to_ids(relations=["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"])[0]
    atom_entity_id = tf.entities_to_ids(entities=["<"+str(atom.iri)+">"])[0]

    #Create the head and tail datasets.
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
    filter_results['still_valid'] = filter_results[["in_training", "in_validation", "in_testing"]].any(axis=1)
    filter_results['score_calibrated'] = filter_results['score'].apply(expit)
    filter_results['score_combined'] = filter_results['score_calibrated']

    return filter_results

def roles(model, k, tf, train, valid, test, atom, target):  
    """
    Predicts the base case for a given role, given a model, parameters, and target. The function returns
    a DataFrame containing the predicted triples and additional information.

    Role pipeline:
        1. Given a role atom r input and a target prediction.
        2.  If target is tails, 
            2.1     We select all the heads which are used in the KG with r. We call this set X.

            2.2     We predict a set Y, where y in Y, and (X, r, y). We keep the top k predictions.
            
        3.  If target is heads, 
            2.1     We select all the tails which are used in the KG with r. We call this set X.

            2.2     We predict a set Y, where y in Y, and (y, r, X). We keep the top k predictions.
            
    Args:
        model: The knowledge graph embedding model.
        k (int): The number of top predictions to consider.
        tf (TriplesFactory): A TriplesFactory instance containing the original triples.
        train: Training set of the knowledge graph.
        valid: Validation set of the knowledge graph.
        test: Test set of the knowledge graph.
        atom: The input role for which the predictions are made.
        target (str): Specifies which part of the triple to predict; either 'head' or 'tail'.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted triples and additional information
                      such as 'score', 'in_training', 'in_validation', 'in_testing', 'still_valid',
                      'score_calibrated', and 'score_combined'.
    """ 

    # TripleFactory creation for the relevant role. If not possible, return an empty DataFrame. 
    try:
        relation = tf.relations_to_ids(relations=["<"+atom.iri+">"])[0]
        new_factory = tf.new_with_restriction(relations=[relation])
    except:
        cols = ['head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label', 'score', 'in_training', 'in_validation', 'in_testing', 'still_valid', 'score_calibrated', 'score_combined']
        return pd.DataFrame(columns=cols)

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
    role_df['score_combined'] = role_df['score_calibrated']

    # We add a column to check if at least one of the three subsets are true. This will be used to verify in the end if each atom prediction in a query has returned a true prediction along the way.
    role_df['still_valid'] = role_df[["in_training", "in_validation", "in_testing"]].any(axis=1)

    # Return the dataframe role_df
    return role_df

def projection(model, k, tf, train, valid, test, input_entities, atom, target):
    """
    Given a model, parameters, and previous prediction results, this function performs a projection,
    predicting the next set of triples. It returns a DataFrame containing the predicted triples and
    additional information.

    Args:
        model:                          The knowledge graph embedding model.
        k (int):                        The number of top predictions to consider.
        tf (TriplesFactory):            A TriplesFactory instance containing the original triples.
        train (TriplesFactory):         Training set of the knowledge graph.
        valid (TriplesFactory):         Validation set of the knowledge graph.
        test (TriplesFactory):          Test set of the knowledge graph.
        input_entities (pd.DataFrame):  The previous prediction results to be used as input for the projection.
        atom:                           The input role for which the predictions are made.
        target (str):                   Specifies which part of the triple to predict; either 'head' or 'tail'.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted triples and additional information
                      such as 'score', 'in_training', 'in_validation', 'in_testing', 'still_valid',
                      'score_calibrated', and 'score_combined'.
    """

    #If the input is empty, return an empty frame.
    if input_entities.empty:
        cols = ['head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label', 'score', 'in_training', 'in_validation', 'in_testing', 'still_valid', 'score_calibrated', 'score_combined']
        return pd.DataFrame(columns=cols)
    try:
        relation = tf.relations_to_ids(relations=["<"+atom.iri+">"])[0]
    except:
        cols = ['head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label', 'score', 'in_training', 'in_validation', 'in_testing', 'still_valid', 'score_calibrated', 'score_combined']
        return pd.DataFrame(columns=cols)
    
    # Create necessary dataset
    if target == 'tail':
        #This variable is used in the final step of this method for identfying duplicates
        drop_dups_on = 'tail_id'
        head_entities = input_entities['entity_id'].to_list()
        dataset = predict.PartiallyRestrictedPredictionDataset(heads=head_entities, relations=relation, target="tail")

    if target == 'head':
        #This variable is used in the final step of this method for identfying duplicates
        drop_dups_on = 'head_id'
        tail_entities = input_entities['entity_id'].to_list()
        dataset = predict.PartiallyRestrictedPredictionDataset(tails=tail_entities, relations=relation, target="head")

    # Start projection prediction and save the results as a DataFrame
    consumer = predict.TopKScoreConsumer(k=k)
    print("\nCurrently predicting projection for role %s.\n" % (atom.name))
    predict.consume_scores(model, dataset, consumer)
    score_pack = consumer.finalize().process(tf).add_membership_columns(training=train, validation=valid, testing=test)
    projection_df = score_pack.df

    #calibrate with the logistic sigmoid function, expit.
    projection_df['still_valid'] = projection_df[["in_training", "in_validation", "in_testing"]].any(axis=1)
    projection_df['score_calibrated'] = projection_df['score'].apply(expit)

    #Combine scores and update still_valid with and-gating the previous atom.
    projection_df['score_combined'], projection_df['still_valid'] = combine_scores(input_entities, projection_df, target)
    projection_df = projection_df.sort_values(['score_combined'], ignore_index=True, ascending=False).drop_duplicates(subset=[drop_dups_on], ignore_index=True, keep='first')
    
    #Return the dataframe
    return projection_df

def intersection(df1, df2):
    """
    Given two DataFrames, this function intersects them based on the common 'entity_id' values,
    retains the top scores, sorts the resulting DataFrame in descending order based on the 
    'score_combined' column, and removes duplicates, keeping the first occurrence with the highest score.
    The latter is equivalent by utilizing the t-conorm "max", as keeping the top score is exactly this.

    Args:
        df1 (pd.DataFrame): The first DataFrame to intersect.
        df2 (pd.DataFrame): The second DataFrame to intersect.

    Returns:
        pd.DataFrame: A DataFrame containing the intersection of df1 and df2, sorted by 'score_combined'
                      in descending order, with duplicates removed.
    """
    
    # Create a new DataFrame containing common 'entity_id' values from both input DataFrames.
    entities = df1[df1['entity_id'].isin(df2['entity_id'])]    

    # Sort the new DataFrame by 'score_combined' in descending order and remove duplicates based
    # on 'entity_id', keeping the last occurrence (lowest score) in the sorted DataFrame.
    entities = entities.sort_values(['score_combined'], ignore_index=True, ascending=False).drop_duplicates(subset=['entity_id'], ignore_index=True, keep='last')

    return entities

def intersection_input_entities(querybody, tf, source_var):
    """
    Given a source variable in the query that is bound, return the intersection. This method is necessary
    for queries where there are no unbound variables, like: q(w):-P22(y,w)^P25(w,y).
    Based on the source variable (not distinguished variable), it will intersect the input. In this example, 
    it would be y.

    Args:
        querybody (QueryBody):  A custom class from PerfectRef with a list of atoms.
        tf (TriplesFactory):    A TriplesFactory instance containing the original triples.
        source_var (string):    The variable name for which we want to intersect.
    
    Returns:
        pd.DataFrame:   A DataFrame solely containing the entity IDs for the intersection; utilized for 
                        producing dataset in later prediction step.
    """
    
    #Create empty Dataframe
    cols = ['entity_id', 'entity_label', 'still_valid', 'origin', 'score_calibrated', 'score_combined']
    entities_df = pd.DataFrame(columns=cols)

    intersect = list()

    # Iterate through every atom, and if it contains the source variable add it to the intersect list
    for atom in querybody.body:
        relation = tf.relations_to_ids(relations=["<"+atom.iri+">"])[0]
        new_factory = tf.new_with_restriction(relations=[relation])
        var1, var2 = atom.var1.original_entry_name, atom.var2.original_entry_name

        # Save the head entities from the KG
        if var1 == source_var:
            intersect.append(np.unique(new_factory.mapped_triples[:, 0], return_index=False))

        #Save the tail entities from the KG
        if var2 == source_var:
            intersect.append(np.unique(new_factory.mapped_triples[:, 2], return_index=False))

    # Intersect the lists of entities.
    entities_df['entity_id'] = reduce(np.intersect1d, intersect)

    # Return the dataframe
    return entities_df

def disjunction(list_of_dfs):
    """
    Given a list of DataFrames, this function takes the union of all DataFrames, retains the top scores,
    sorts the resulting DataFrame in descending order based on the 'score_combined' column, and removes
    duplicates, keeping the first occurrence with the highest score.

    Args:
        list_of_dfs (list of pd.DataFrame): A list of DataFrames to be combined.

    Returns:
        pd.DataFrame: A DataFrame containing the union of all input DataFrames, sorted by 'score_combined'
                      in descending order, with duplicates removed.
    """
    
    # Concatenate the rows from all input DataFrames.
    entities = pd.concat(list_of_dfs, axis=0, ignore_index=True)

    # Sort the new DataFrame by 'score_combined' in descending order and remove duplicates based
    # on 'entity_id', keeping the first occurrence (highest score) in the sorted DataFrame.
    entities = entities.sort_values(['score_combined'], ignore_index=True, ascending=False).drop_duplicates(subset=['entity_id'], keep='first', ignore_index=True)

    return entities

def get_base_case(base_cases, current_config):
    """
    Retrieves the base case results corresponding to the given configuration from a list of
    existing base cases.

    Args:
        base_cases (list):      A list of dictionaries containing existing base case configurations
                                and results.
        current_config (dict):  The configuration for which the base case results are to be
                                retrieved.

    Returns:
        pd.DataFrame:           A deep copy of the base case results DataFrame corresponding to the
                                given configuration, or None if the base case is not found.
    """

    # Iterate through the base_cases list.
    for b in base_cases:

        # If the current_config matches a base case configuration, return a deep copy of the
        # corresponding base case results DataFrame.
        if b['config'] == current_config:
            return deepcopy(b['entities_df'])

    # If the base case is not found, print an error message and return None.
    raise RuntimeError("Did not find this base_case. Something wrong happened at the previous base case prediction step.")

def new_base_case(base_cases, config):
    """
    Given a list of base cases and a configuration, this function checks if the configuration
    already exists in the base cases. If it does, the function returns False, indicating that
    the base case has already been predicted. Otherwise, it returns True.

    This is handy, so we dont do duplicate predictions.

    Args:
        base_cases (list of dict): A list of dictionaries containing base case configurations
                                    and their results.
        config (dict): The configuration to check against the base cases.

    Returns:
        bool: True if the configuration is not in the base cases, indicating it's a new base case;
              False otherwise.
    """
    
    # Iterate through each base case in the base_cases list.
    for b in base_cases:
        # If the configuration matches an existing base case configuration, return False.
        if b['config'] == config:
            return False

    # If no match is found, return True, indicating it's a new base case.
    return True

def base_predictions(query, base_cases, base_case_path, dataset, model, model_params, k, tf, train, valid, test, aboxpath):
    """

    Handles base case predictions, which are cases where there is a concept or a role with an
    unbound variable in the query. For each atom in the query, this function checks if it's a
    new base case present and performs predictions accordingly.

    Args:
        query (Query):          The query object containing the body with atoms.
        base_cases (list):      A list of dictionaries containing existing base case configurations and results.
        base_case_path (str):   The path to the saved base case pickle file.
        dataset (str):          The name of the dataset used for the predictions.
        model (BaseModel):      The pre-trained model for predictions.
        model_params (dict):    Model parameters.
        k (int):                The number of top predictions to consider.
        tf (TripleFactory): T   ripleFactory object for mapping and unmapping entity/relation IDs.
        train (pd.DataFrame):   Training set DataFrame.
        valid (pd.DataFrame):   Validation set DataFrame.
        test (pd.DataFrame):    Test set DataFrame.
        aboxpath (str):         The path to the ABox file.

    Returns:
        list:                   An updated list of base case configurations and results.
    
    """

    # Iterate through each atom in the query body.
    for g in query.body:

        is_base_case = False

        # Handle concept base case predictions.
        if isinstance(g, AtomConcept):

            # If the concept variable is bound (which should always be the case).
            if g.var1.bound:

                current_config = {'dataset': dataset, 'model': model_params, 'k': k, 'iri': g.iri, 'target': 'head'}
                is_base_case = True

                # If the current configuration is a new base case, perform concept predictions.
                if new_base_case(base_cases, current_config):
                    predicted_entities_df = concepts(dataset, model, k, tf, train, valid, test, g, aboxpath)
                    
        # Handle role base case predictions.
        if isinstance(g, AtomRole):

            # If there is an unbound variable and the target is 'tail'.
            if g.var1.unbound and g.var2.bound:
                
                current_config = {'dataset': dataset, 'model': model_params, 'k': k, 'iri': g.iri, 'target': 'tail'}
                is_base_case = True

                # If the current configuration is a new base case, perform role predictions.
                if new_base_case(base_cases, current_config):
                    predicted_entities_df = roles(model, k, tf, train, valid, test, g, current_config['target'])
                    
            # If there is an unbound variable and the target is 'head'.
            if g.var2.unbound and g.var1.bound:

                current_config = {'dataset': dataset, 'model': model_params, 'k': k, 'iri': g.iri, 'target': 'head'}
                is_base_case = True

                # If the current configuration is a new base case, perform role predictions.
                if new_base_case(base_cases, current_config):
                    predicted_entities_df = roles(model, k, tf, train, valid, test, g, current_config['target'])
                    
        
        # Update the list of base cases if the current configuration is a new base case.
        # Also, write to file such that it will be remembered next time code is runned.
        if is_base_case:
            if new_base_case(base_cases, current_config):
                base_cases.append({'config': current_config, 'entities_df': predicted_entities_df})
                update_prediction_pickle(base_cases, base_case_path)

    return base_cases

def standardize_dataframe(df, target, query_string):

    """
    Standardizes a dataframe by removing head_id, tail_ids and substituting them with entity_id.

    Args:
        df (pd.DataFrame): The input dataframe to be standardized.
        target (str): The target variable, either 'head' or 'tail'.
        query_string (str): The query string to be added as 'origin' to the standardized dataframe.

    Returns:
        pd.DataFrame: The standardized dataframe with the specified columns.
    """

    # Add the origin column with the query string value.
    df['origin'] = query_string

    # Specify the columns to be returned in the standardized dataframe.
    returning_columns = ['entity_id', 'entity_label', 'still_valid', 'origin', 'score_calibrated', 'score_combined']

    # Remove dependency of target knowledge by creating a mapping of column names.
    if target == 'tail':
        renaming = {'tail_id': 'entity_id', 'tail_label': 'entity_label'}
    if target == 'head':
        renaming = {'head_id': 'entity_id', 'head_label': 'entity_label'}

    # Rename the columns in the dataframe based on the target.
    df = df.rename(columns=renaming)

    # Sort the dataframe by score_combined and remove duplicates.
    df = df.sort_values(['score_combined'], ignore_index=True, ascending=False).drop_duplicates(subset=['entity_id'], ignore_index=True, keep='first')

    # Return the standardized dataframe with the specified columns.
    return df[returning_columns]

def combine_scores(previous_atom, current_atom, target):
    """
    Combine the scores of previous and current atoms and update the validity.
    The score combination is achieved by mapping the entities to the score.
    After this, we perform the t-norm on the scores, and a and-gate on the
    "still_valid" column. The still valid column highlights whether a prediction return
    in the end has been True for every atom along the way.

    Args:
        previous_atom (pd.DataFrame): The dataframe containing the scores and validity of the previous atom.
        current_atom (pd.DataFrame): The dataframe containing the scores and validity of the current atom.
        target (str): The target variable, either 'head' or 'tail'.

    Returns:
        tuple: A tuple containing two pandas Series objects:
            - The combined scores of the previous and current atoms using the t-norm function.
            - The updated validity based on both atoms
    """

    #Calculation for "still_valid".
    mapping_valid = previous_atom.set_index('entity_id')['still_valid']

    #Mapping for scores
    mapping_scores = previous_atom.set_index('entity_id')['score_combined']
    
    # For target 'tail': e(x), r1(x, y)
    if target == 'tail':
        scores = current_atom.apply(lambda x: t_norm(x['score_calibrated'], mapping_scores.get(x['head_id'], 1)), axis=1)
        validity = current_atom.apply(lambda x: and_gate(x['still_valid'], mapping_valid.get(x['head_id'], 1)), axis=1)

    # For target 'head': e(x), r1(y, x)
    if target == 'head':
        scores = current_atom.apply(lambda x: t_norm(x['score_calibrated'], mapping_scores.get(x['tail_id'], 1)), axis=1)
        validity = current_atom.apply(lambda x: and_gate(x['still_valid'], mapping_valid.get(x['tail_id'], 1)), axis=1)

    return scores, validity

def query_pipeline(query, base_cases, base_cases_path, enable_online_lookup, dataset, model, model_params, k, tf, train, valid, test, aboxpath, n):
    """
    Execute the main query pipeline for the input query. This pipeline consists of preprocessing, prediction, and result aggregation steps.

    Preprocessing:
        1. Save the query's distinguished variables.
        2. Build a variable hierarchy which will determine the prediction order of the atoms.
        3. Sort the query's atoms based on the variable structure.

    Prediction:
        1. Predict all base cases used in the query.
        2. Predict the rest in the sorted order, and if a target already exists, intersect the results.

    Result Aggregation:
        3. Return the entities dataframe from the entities dictionary where the dictionary key is the distinguished variable.

    Args:
        query (dict): The high-level dictionary of the current query that will be predicted.
        base_cases (dict): The dictionary containing all the base cases used in the query.
        base_cases_path (str): The path to the base cases file.
        enable_online_lookup (bool): A flag to enable or disable online KG lookups.
        dataset (str): The dataset to be used for the query.
        model (PyKEEN Class): The model to be used for the query.
        model_params (dict): The parameters for the model.
        k (int): The number of top predictions to consider.
        tf (TriplesFactory): The type of transitive function to use.
        train (TriplesFactory): The path to the train dataset.
        valid (TriplesFactory): The path to the valid dataset.
        test (TriplesFactory): The path to the test dataset.
        aboxpath (str): The path to the ABox file.
        n (int): The number of top predictions to use for computing metrics - hits@n.

    Returns:
        tuple: A tuple containing the final dataframe and the result metrics.
    """
    # Identify the distinguished variable
    distinguished_variable = query['q1'].head.var1.original_entry_name

    # Variable hierarchy and query atom sorting
    for querybody in query['rewritings']:
        variable_hierarchy = build_variable_hierarchy(querybody, distinguished_variable)
        querybody.body = sort_atoms_by_depth(variable_hierarchy, querybody)
        querybody.variable_hierarchy = variable_hierarchy
    
    # Base case predictions
    for querybody in query['rewritings']:
            base_cases = base_predictions(querybody, base_cases, base_cases_path, dataset, model, model_params, k, tf, train, valid, test, aboxpath)

    # Main prediction loop
    for idx, querybody in enumerate(query['rewritings']):
        
        # Dictionary for keeping track of entities per variable.
        entities = dict()
        query_string = get_query_string(querybody, distinguished_variable)
        print("Predicting reformulation %i / %i. \nCurrent reformulation: %s\n\n Please wait ..." % (idx + 1, len(query['rewritings']), query_string))

        # 1. BASE CASES
        ## Bind base-predictions to variables and standardize the dataframe
        for atom in querybody.body:
            if isinstance(atom, AtomConcept):
                if atom.var1.bound:
                    target = 'head'
                    current_config = {'dataset': dataset, 'model': model_params, 'k': k, 'iri': atom.iri, 'target': target}

                    # If the target variable already has data from a previous atom
                    intersect = True if atom.var1.original_entry_name in entities else False

                    #Get the base case from the previous step
                    entities_df = get_base_case(base_cases, current_config)
                    
                    if intersect:
                        entities[atom.var1.original_entry_name] = intersection(entities[atom.var1.original_entry_name],standardize_dataframe(entities_df, target, query_string))
                    else:
                        entities[atom.var1.original_entry_name] = standardize_dataframe(entities_df, target, query_string)

            if isinstance(atom, AtomRole):
        
                # If basecase, and target is tail
                if atom.var1.unbound and atom.var2.bound:
                    target = 'tail'
                    #The current configuration for the atom
                    current_config = {'dataset': dataset, 'model': model_params, 'k': k, 'iri': atom.iri, 'target': target}
                    
                    # If the target variable already has data from a previous atom
                    intersect = True if atom.var2.original_entry_name in entities else False

                    #Extract the base case from the generalization, and bind it to
                    # the entities dictionary where the key is the target variable.
                    # If the target variable already has entities, we intersect. 
                    # We also standardize the dataframe, such that we remove head and tail specific columns.
                    entities_df = get_base_case(base_cases, current_config)
                    
                    
                    if intersect:
                        entities[atom.var2.original_entry_name] = intersection(entities[atom.var2.original_entry_name],standardize_dataframe(entities_df, target, query_string))
                    else:
                        entities[atom.var2.original_entry_name] = standardize_dataframe(entities_df, target, query_string)

                # If basecase, and target is head
                if atom.var1.bound and atom.var2.unbound:
                    target = 'head'
                    current_config = {'dataset': dataset, 'model': model_params, 'k': k, 'iri': atom.iri, 'target': target}

                    # If the target variable already has data from a previous atom
                    intersect = True if atom.var1.original_entry_name in entities else False

                    #Extract the base case from the generalization, and bind it to
                    # the entities dictionary where the key is the target variable.
                    # We also standardize the dataframe, such that we remove head and tail specific columns.
                    entities_df = get_base_case(base_cases, current_config)

                    if intersect:
                        entities[atom.var1.original_entry_name] = intersection(entities[atom.var1.original_entry_name],standardize_dataframe(entities_df, target, query_string))
                    else:
                        entities[atom.var1.original_entry_name] = standardize_dataframe(entities_df, target, query_string)

        # PROJECTIONS 
        for atom in querybody.body:
            
            # This will only apply to roles
            if isinstance(atom, AtomRole):

                #Pure projection atom
                if (atom.var1.bound and atom.var2.bound):
                    
                    target, target_var, source_var = get_target_variable(querybody.variable_hierarchy, atom.var1.original_entry_name, atom.var2.original_entry_name)
                    
                    if source_var in entities:
                        input_entities = entities[source_var]
                    else:
                        #There are no unbounds, and hence we need to combine the highest depth variable
                        entities[source_var] = intersection_input_entities(querybody, tf, source_var)
                        input_entities = entities[source_var]

                    # If the target variable already has data from a previous atom
                    # The intersect variable is unneccessary but created for readability
                    intersect = True if target_var in entities else False

                    if intersect:
                        entities[target_var] = intersection(entities[target_var],standardize_dataframe(projection(model, k, tf, train, valid, test, input_entities, atom, target), target, query_string))
                    else:
                        entities[target_var] = standardize_dataframe(projection(model, k, tf, train, valid, test, input_entities, atom, target), target, query_string)

        # Save the answer DF
        # Check whether the predictions are in the KG
        if not entities[distinguished_variable].empty:
            entities[distinguished_variable]['local_kg_hit_rewriting'] = is_prediction_kg_hit(entities[distinguished_variable], query['kglookup']) if query['kglookup'] else False
            entities[distinguished_variable]['online_kg_hit_rewriting'] = online_kg_lookup(entities[distinguished_variable], querybody, dataset) if enable_online_lookup else False
        else:
            entities[distinguished_variable]['local_kg_hit_rewriting'] = None
            entities[distinguished_variable]['online_kg_hit_rewriting'] = None

        querybody.answer = (entities[distinguished_variable])
    

    # Take the union of the results
    list_of_final_dfs = list()

    for querybody in query['rewritings']:
        list_of_final_dfs.append(querybody.answer)

    # Take the union of off rewritings and keep the best scoring one.
    final_df = disjunction(list_of_final_dfs)

    if not final_df.empty:
        # Check if final entities are present in the KG-answer for the original query.

        final_df['local_kg_hit_original'] = is_prediction_kg_hit(final_df, query['kglookup']) if query['kglookup'] else False
        final_df['online_kg_hit_original'] = online_kg_lookup(final_df, query, dataset) if enable_online_lookup else False
    else:
        final_df['local_kg_hit_original'] = False
        final_df['online_kg_hit_original'] = False
    

    results_metrices = {f'hits@{n}': compute_hits_at_k(final_df, n), 'mrr':compute_mrr(final_df)}
    return final_df, results_metrices

def predict_parsed_queries(all_queries, base_cases, base_cases_path, enable_online_lookup, dataset, model, model_params, k, tf, train, valid, test, tbox_ontology, aboxpath, result_path, n):
    """
    Predict answers for parsed queries, save results and metrics to files, and compute average metrics for each query structure.
    In method prediction hierarchy, this method is on top.

    Args:
        all_queries (dict): Dictionary containing all parsed queries, grouped by query structure.
        base_cases (dict): Dictionary containing all base cases used in the queries.
        base_cases_path (str): Path to the base cases file.
        enable_online_lookup (bool): Flag to enable/disable online KG lookups.
        dataset (str): Dataset to be used for the queries.
        model (PyKEEN Class): Model to be used for the queries.
        model_params (dict): Parameters for the model.
        k (int): Number of top predictions to consider.
        tf (TriplesFactory): Type of transitive function to use.
        train (TriplesFactory): Path to the train dataset.
        valid (TriplesFactory): Path to the valid dataset.
        test (TriplesFactory): Path to the test dataset.
        tbox_ontology (str): Path to the TBox ontology.
        aboxpath (str): Path to the ABox file.
        result_path (str): Path to the directory where results will be saved.
        n (int): Number of top predictions to use for computing metrics - hits@n.

    Returns:
        None. Results are written to files.
    """

    # Iterate through each query structure type
    for query_structure in all_queries.keys():
        structure_metrices = list()

        # Get the answer for all queries in each structure
        for idx, query in enumerate(all_queries[query_structure]):
            
            # Execute the query pipeline and retrieve results and metrics
            query['answer'], results_metrics = query_pipeline(query, base_cases, base_cases_path, enable_online_lookup, dataset, model, model_params, k, tf, train, valid, test, aboxpath, n)

            # Save the answer to a JSON file
            query['answer'].to_json(f"{result_path}/every_structure/{query_structure}-{idx+1}.json", indent=4, orient = 'index')

            # Save metrics to a JSON file
            with open(f'{result_path}/every_structure/{query_structure}-{idx+1}-results.json', 'w') as fp:
                json.dump(results_metrics, fp, indent=4)

            structure_metrices.append(results_metrics)

        # Calculate average metrics for each query structure
        final_results = average_metrics(structure_metrices, n)
        final_results['query_structure'] = query_structure
        final_results['number_of_queries'] = len(all_queries[query_structure])
        final_results['model'] = model_params
        final_results['dataset'] = dataset
        final_results['cutoff_value'] = k

        # Save the final results to a JSON file
        with open(f'{result_path}/{dataset}-{ model_params["selected_model_name"]}-dim{model_params["dim"]}-epoch{model_params["epoch"]}-k{k}-numbofqueries:{final_results["number_of_queries"]}-{query_structure}-final_results.json', 'w') as fp:
                json.dump(final_results, fp, indent=4)

    return None


