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
tbox_import_file = "family_wikidata.owl"
dataset = "family"
project_name = "001"
transductive_models = ["TransE", "BoxE", "RotatE", "DistMult", "CompGCN"]
current_model = None
parsed_generated_queries = None
number_of_queries_per_structure = 25
queries_from_generation = None
enable_online_lookup = True
current_model_params = {'selected_model_name': None, 'dim': None, 'epoch': None}
t_box_path = "dataset/"+dataset+"/tbox/"+tbox_import_file
a_box_path = "dataset/"+dataset+"/abox/transductive/"
tbox_ontology = load_ontology(t_box_path)
tf = TriplesFactory.from_path(a_box_path + "all.txt")
train, valid, test = tf.split([0.8, 0.1, 0.1], random_state=RANDOM_SEED, method="cleanup")
base_cases_path  = "testcases/" + project_name + "/base_cases.pickle"

#Load previous base predictions
if os.path.exists(base_cases_path):
    with open(base_cases_path, 'rb') as file:
        base_cases = pickle.load(file)
else:
    base_cases = list()

query_path = "testcases/" + project_name + "/queries/" + dataset + "/queries_k-" + str(number_of_queries_per_structure) + ".txt"

parsed_generated_queries = dict()
unloaded = True

try:
    with open(query_path) as json_file:
        queries_from_generation = json.load(json_file)
    unloaded = False
except:
    print("Doesnt exist. Try again.")


for key in queries_from_generation.keys():
    parsed_generated_queries[key] = list()
    for q in queries_from_generation[key]:
        parsed_generated_queries[key].append(query_structure_parsing(q, key, tbox_ontology))

#do local KG-lookup
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

for i in range(1,6):
    if i == 1:
        current_model = "TransE"
        dim = 192
        epoch = 24
    elif i == 2:
        current_model = "BoxE"
        dim = 192
        epoch = 24
    elif i == 3:
        current_model = "RotatE"
        dim = 192
        epoch = 24
    elif i == 4:
        current_model = "DistMult"
        dim = 192
        epoch = 16
    elif i == 5:
        current_model = "CompGCN"
        dim = 192
        epoch = 24
    else:
        current_model = None
    
    # Set up the configuration file path for the current model
    current_config = "testcases/" + project_name + "/models/dataset:" + dataset + "_model:" + current_model + "_dim:" + str(dim) + "_epoch:" + str(epoch)

    if os.path.exists(current_config):
        # If the configuration file exists, load the trained model and return it along with the selected model name, dimension, and number of epochs
        model_path = current_config + "/trained_model.pkl"
        current_model_params = {'selected_model_name': current_model, 'dim': dim, 'epoch': epoch}
        current_model = load(model_path)


    pth = "testcases/" + project_name + "/queries/" + dataset + "/"
    filename = "queries" + "_k-" + str(number_of_queries_per_structure) + "_parsed_and_rewritten.pickle"
    full_pth = pth + filename

    #Reformulate
    parsed_generated_queries = query_reformulate(parsed_generated_queries, rewriting_upper_limit, full_pth, t_box_path)

    #Reformulation KG Lookup
    parsed_generated_queries = kg_lookup_rewriting(parsed_generated_queries, dataset, a_box_path, tf, full_pth) 

    #Predict
    predict_parsed_queries(parsed_generated_queries,base_cases, base_cases_path, enable_online_lookup, dataset, current_model, current_model_params, k, tf, train, valid, test, tbox_ontology, a_box_path, result_path, n)

