import os

from pykeen.models import *
from pykeen.pipeline import pipeline
from torch import load

from utilities import *

def training(seed, transductive_models, project_name, dataset, train, valid, test):
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
                random_seed=seed,
                model=current_model,
                model_kwargs=dict(embedding_dim=dim),
                epochs=epoch,
                dimensions=dim,
                device=get_device(),
            )

            results.save_to_directory(current_config)

    return None

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