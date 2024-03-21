import errno

import numpy as np
import pickle
import shap
import tensorflow as tf
from tensorflow.keras.saving import load_model
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass


@dataclass
class Explainer:
    results_path: str
    llms: list[str]

    def produce_shap_plots(self, 
                           strats,
                           plot_type, 
                           num_background_points=100,
                           seed=0,
                           max_ngram_display=10,
                           show=False):
        with open(f'{self.results_path}/x_train_dict.pickle', 'rb') as handle:
            x_train_dict = pickle.load(handle)

        with open(f'{self.results_path}/x_val_dict.pickle', 'rb') as handle:
            x_val_dict = pickle.load(handle)

        with open(f'{self.results_path}/x_test_dict.pickle', 'rb') as handle:
            x_test_dict = pickle.load(handle)

        with open(f'{self.results_path}/ngrams_dict.pickle', 'rb') as handle:
            ngrams_dict = pickle.load(handle)

        with open(f'{self.results_path}/train_dict.pickle', 'rb') as handle:
            train_dict = pickle.load(handle)

        with open(f'{self.results_path}/val_dict.pickle', 'rb') as handle:
            val_dict = pickle.load(handle)

        with open(f'{self.results_path}/test_dict.pickle', 'rb') as handle:
            test_dict = pickle.load(handle)

        for llm in self.llms:
            print(llm)

            for stratum_name in strats:
                print(stratum_name)
                stratum_levels = set(test_dict[llm][stratum_name])
                num_cols = 3
                num_rows = (len(stratum_levels) - 1) // 3 + 1

                # Initialise a figure env (cannot use fig because shap lib does not return axes; instead does the
                # plotting)
                plt.figure(figsize=(num_cols * 5, num_rows * 5))
                for ix, stratum in enumerate(stratum_levels):
                    print(stratum)
                    # Get the specific stratification mask
                    rung_mask = [test_dict[llm][stratum_name][j] == stratum for j in
                                 test_dict[llm][stratum_name].keys()]

                    # Load the model
                    model = load_model(f"{self.results_path}/{llm}/model.h5")

                    # Extract the llm-specific data
                    x_train = x_train_dict[llm]
                    x_test = x_test_dict[llm][rung_mask]
                    ngrams = ngrams_dict[llm]

                    np.random.seed(seed)
                    num_rand_idx = num_background_points
                    random_indices = np.random.choice(x_train.shape[0], size=num_rand_idx, replace=False)
                    random_training_samples = x_train[random_indices, :]

                    shap_explainer = shap.DeepExplainer(model, data=random_training_samples)
                    shap_values = shap_explainer.shap_values(x_test)
                    base_value = float(shap_explainer.expected_value[0])
                    explanation = shap.Explanation(values=shap_values[:, :, 0],
                                                   base_values=[base_value] * len(shap_values),
                                                   # See notion/the_various_shap_plots
                                                   data=x_test,
                                                   feature_names=ngrams)

                    plt.subplot(num_rows, num_cols, ix + 1)

                    # SUMMARY PLOT
                    if plot_type == "summary":
                        shap.summary_plot(shap_values[:, :, 0],
                                          x_test,
                                          feature_names=ngrams,
                                          plot_type="bar",
                                          show=False,
                                          plot_size=None,  # To be able to plot the subplots next to one another
                                          )

                    # VIOLIN PLOT
                    if plot_type == "violin":
                        shap.plots.violin(explanation,
                                          plot_size=None,
                                          max_display=max_ngram_display,
                                          show=False,
                                          )

                    plt.title(f"{llm} {stratum_name}: {stratum}")
                plt.tight_layout()

                try:
                    plot_dir = f"{self.results_path}/{llm}/{stratum_name}/"
                    os.mkdir(plot_dir)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                plt.savefig(fname=f"{plot_dir}/{plot_type}-{stratum_name}.png")

                if show:
                    plt.show()


if __name__ == "__main__":
    results_path = "results/dummy"
    llms = ["gpt3.04",]
    strats = ["rung"]
    plot_type = "violin"

    explainer = Explainer(results_path=results_path,
                          llms=llms, )

    explainer.produce_shap_plots(strats=strats,
                                 plot_type=plot_type,
                                 num_background_points=200,
                                 seed=10,
                                 max_ngram_display=20)

    print("Exit ok")
