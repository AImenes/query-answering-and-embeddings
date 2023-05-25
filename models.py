import os

from pykeen.models import *
from pykeen.pipeline import pipeline
from torch import load

from utilities import *

def training(seed, project_name, dataset, train, valid, test):
    """
    Train transductive models with the given parameters.

    Args:
        seed (int): The random seed to use for the training.
        transductive_models (list): A list of transductive models to train.
        project_name (str): The name of the test case, default 001.
        dataset (str): The name of the dataset to use for the training.
        train (TriplesFactory): The training dataset.
        valid (TriplesFactory): The validation dataset.
        test (TriplesFactory): The test dataset.

    Returns:
        None. Writes to file.
    """
    # Train each transductive model with the given parameters
    
    models = {
    '1': "BoxE",
    '2': "CompGCN",
    '3': "DistMult",
    '4': "RotatE",
    '5': "TransE"
    }

    for entry in models.keys():
        print(" | "+entry+":\t", models[entry])

    selection = None
    while selection not in models.keys():
        selection = input("\nPlease Select a model [1-5]: ")
    
    if selection == '1':
        current_model = "BoxE"
    elif selection == '2':
        current_model = "CompGCN"
    elif selection == '3':
        current_model = "DistMult"
    elif selection == '4':
        current_model = "RotatE"
    elif selection == '5':
        current_model = "TransE"
    else:
        current_model = selection

    # Get the dimension and number of epochs for the current model from user input
    dim = int(input("Enter dimension for model %s: " % (current_model)))
    epoch = int(input("Enter number of epochs for model %s: " % (current_model)))

    # Set up the configuration file path for the current model
    current_config = "testcases/" + project_name + "/models/dataset:" + dataset + "_model:" + current_model + "_dim:" + str(dim) + "_epoch:" + str(epoch)

    if os.path.exists(current_config):
        # If the configuration file already exists, print an error message and do not train the model again
        print("Configuration already trained. Edit configuration or delete model directory.")
    else:
        # Train the model with the given parameters
        results = pipeline(
            training_loop='sLCWA',
            training=train,
            validation=valid,
            testing=test,
            random_seed=seed,
            model=current_model,
            model_kwargs=dict(embedding_dim=dim),
            epochs=epoch,
            dimensions=dim,
            device=get_device(),
        )

        # Save the trained model to the configuration file path
        results.save_to_directory(current_config)
print("Training finished.")

def load_model(project_name, dataset):
    """
    Load a trained model for the given project and dataset.

    Args:
        project_name (str): The name of the project.
        dataset (str): The name of the dataset.

    Returns:
        The trained model and a dictionary containing its selected model name, dimension, and number of epochs if the configuration file exists, otherwise None.
    """
    # Get the model type, dimension, and number of epochs from user input
    model_number = int(input("Enter 1 for TransE\tEnter 2 for BoxE\tEnter 3 RotatE\tEnter 4 DistMult\tEnter 5 CompGCN\t(1-5): "))
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
    epochs = int(input("Enter number of epochs for model %s: " % (current_model)))

    # Set up the configuration file path for the current model
    current_config = "testcases/" + project_name + "/models/dataset:" + dataset + "_model:" + current_model + "_dim:" + str(dim) + "_epoch:" + str(epochs)

    if os.path.exists(current_config):
        # If the configuration file exists, load the trained model and return it along with the selected model name, dimension, and number of epochs
        model_path = current_config + "/trained_model.pkl"
        trained_model = load(model_path)
        print("model loaded successfully")
        return trained_model, {'selected_model_name': current_model, 'dim': dim, 'epoch': epochs}

    # If the configuration file does not exist, return None
    return None