import os
from datetime import datetime

import yaml

from src.preprocessor import Preprocessor
from src.trainer import train_ngram_model
from src.explainer import Explainer


def main(path_to_config):
    # Load the configuration parameters
    PATH_TO_CONFIG = path_to_config

    with open(PATH_TO_CONFIG) as stream:
        try:
            PARAMS = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Create a timestamped results folder for this particular run and update PARAMS accordingly
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_path = f"{PARAMS['general']['results_path']}/{now}"
    os.mkdir(results_path)
    PARAMS['general']['results_path'] += f"/{now}"

    # Save the config.yaml in the new results folder
    with open(f"{results_path}/config.yaml", "w") as file:
        yaml.safe_dump(PARAMS, file)

    # Load and preprocess the dataset
    preprocessor = Preprocessor(**PARAMS["general"], **PARAMS["preprocessing"])
    preprocessor.preprocess_dataset()

    # Train the MLPs
    train_ngram_model(x_train_dict=preprocessor.x_train_dict,
                      x_val_dict=preprocessor.x_val_dict,
                      x_test_dict=preprocessor.x_test_dict,
                      train_labels_dict=preprocessor.train_labels_dict,
                      val_labels_dict=preprocessor.val_labels_dict,
                      **PARAMS["training"],
                      **PARAMS["general"], )

    # return results_path

    # Explain the MLPs by producing shap value plots
    explainer = Explainer(results_path=results_path,
                          llms=PARAMS["general"]["llms"],)

    explainer.produce_shap_plots(**PARAMS["explaining"])


if __name__ == "__main__":
    main(path_to_config="conf/config.yaml")
