import os
from datetime import datetime

import yaml

from src.preprocessor import Preprocessor
from src.trainer import Trainer
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

    instructions = PARAMS["instructions"]

    # TODO: So far, cannot do training without preprocessing (see training section below which relies on preprocessor
    #  attributes. Must decide how to either decouple preprocessing and training or treat them as a single block
    if instructions["do_preprocessing"]:
        # Load and preprocess the dataset
        preprocessor = Preprocessor(**PARAMS["general"], **PARAMS["preprocessing"])
        preprocessor.preprocess_dataset()

    if instructions["do_training"]:
        # Train the MLPs
        trainer = Trainer(**PARAMS["general"], **PARAMS["training"])
        train_acc_dict, val_acc_dict, y_pred_dict = trainer.train_ngram_model(x_train_dict=preprocessor.x_train_dict,
                                                                              x_val_dict=preprocessor.x_val_dict,
                                                                              x_test_dict=preprocessor.x_test_dict,
                                                                              train_labels_dict=preprocessor.train_labels_dict,
                                                                              val_labels_dict=preprocessor.val_labels_dict, )

        trainer.produce_all_training_plots(test_dict=preprocessor.test_dict,
                                           history_acc_dict=train_acc_dict,
                                           history_val_acc_dict=val_acc_dict,
                                           y_pred_dict=y_pred_dict, )

    if instructions["do_explaining"]:
        # Explain the MLPs by producing shap value plots
        explainer = Explainer(results_path=results_path,
                              llms=PARAMS["general"]["llms"], )

        explainer.produce_stratified_shap_plots_with_multiple_llms(**PARAMS["explaining"])

        explainer.produce_unstratified_shap_plots(plot_type=PARAMS["explaining"].plot_type,
                                                  num_background_points=PARAMS["explaining"].num_background_points,
                                                  seed=PARAMS["explaining"].seed,
                                                  max_ngram_display=PARAMS["explaining"].max_ngram_display, )


if __name__ == "__main__":
    main(path_to_config="conf/cladder_cleaned.yaml")