import errno
import pickle
import os

import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras.saving import load_model
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Explainer:
    results_path: str
    llms: list[str]

    def produce_unstratified_shap_plots(self,
                                        plot_type,
                                        num_background_points=200,
                                        seed=0,
                                        max_ngram_display=10,
                                        show=False,
                                        test_slice=slice(None, None, None)):

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

        num_cols = 4
        num_rows = 2

        # Initialise a figure env (cannot use fig because shap lib does not return axes; instead does the
        # plotting)
        plt.figure(figsize=(num_cols * 5, num_rows * 5))

        for ix, llm in enumerate(self.llms):
            print(llm)

            # Load the model
            model = load_model(f"{self.results_path}/{llm}/model.h5")

            # Extract the llm-specific data
            x_train = x_train_dict[llm]
            x_test = x_test_dict[llm][test_slice]
            ngrams = ngrams_dict[llm]

            np.random.seed(seed)
            num_rand_idx = num_background_points
            random_indices = np.random.choice(x_train.shape[0], size=num_rand_idx, replace=False)
            random_training_samples = x_train[random_indices, :]

            print()

            shap_explainer = shap.DeepExplainer(model, data=random_training_samples)
            shap_values = shap_explainer.shap_values(x_test)
            base_value = float(shap_explainer.expected_value[0])
            explanation = shap.Explanation(values=shap_values[:, :, 0],
                                           base_values=[base_value] * len(shap_values),
                                           # See notion/the_various_shap_plots
                                           data=x_test,
                                           feature_names=ngrams)

            plt.subplot(num_rows, num_cols, ix + 1)

            if plot_type == "summary":
                shap.summary_plot(shap_values[:, :, 0],
                                  x_test,
                                  feature_names=ngrams,
                                  plot_type="bar",
                                  show=False,
                                  plot_size=None,  # To be able to plot the subplots next to one another
                                  )

            if plot_type == "violin":
                shap.plots.violin(explanation,
                                  plot_size=None,
                                  max_display=max_ngram_display,
                                  show=False,
                                  )

            if plot_type == "beeswarm":
                shap.plots.beeswarm(explanation,
                                    plot_size=None,
                                    max_display=max_ngram_display,
                                    show=False)

            if plot_type == "waterfall":
                shap.plots.waterfall(explanation[0],
                                     # plot_size=None,
                                     # max_display=max_ngram_display,
                                     show=False,
                                     )

            plt.title(f"{llm}")
        plt.tight_layout()

        try:
            plot_dir = f"{self.results_path}/general"
            os.mkdir(plot_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        plt.savefig(fname=f"{plot_dir}/{plot_type}.png")

        if show:
            plt.show()

    def produce_stratified_shap_plots(self,
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

                num_cols = len(stratum_levels)
                num_rows = 1

                # For squarer look
                # num_cols = 3
                # num_rows = (len(stratum_levels) - 1) // 3 + 1

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

    def produce_stratified_shap_plots_with_multiple_llms(self,
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

        for stratum_name in strats:
            print(stratum_name)
            # Note: we are assuming that all LLMs have the same strata (i.e. that there are enough data points
            # in the test set to span all categories)
            stratum_levels = set(test_dict[self.llms[0]][stratum_name])
            num_cols = len(stratum_levels)
            num_rows = len(self.llms)
            subplot_ix = -1

            # Initialise a figure env (cannot use fig because shap lib does not return axes; instead does the plotting)
            plt.figure(figsize=(num_cols * 5, num_rows * 5))

            for ix_llm, llm in enumerate(self.llms):
                print(llm)

                for ix, stratum in enumerate(stratum_levels):
                    subplot_ix += 1
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

                    plt.subplot(num_rows, num_cols, subplot_ix+1)

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
                plot_dir = f"{self.results_path}/general/"
                os.mkdir(plot_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            plt.savefig(fname=f"{plot_dir}/{stratum_name}.png")

            if show:
                plt.show()

    @staticmethod
    def _analyse_tf_idf_value_distribution_in(arr: np.ndarray[np.ndarray],
                                              ngrams: np.ndarray[float],
                                              ngram: str,
                                              ) -> None:
        # Note: arr is an array of TF-IDF arrays. The TF-IDF arrays are of the same length as the ngrams array
        ngram_tf_idf = arr[:, ngrams == ngram]
        ngram_tf_idf = ngram_tf_idf[ngram_tf_idf != 0]

        if ngram_tf_idf.size != 0:
            print("max: ", max_val := max(ngram_tf_idf))
            print("min: ", min_val := min(ngram_tf_idf))
            print("range: ", max_val - min_val)

            fig, ax = plt.subplots()
            ax.hist(ngram_tf_idf, bins=20)
            plt.show()
        else:
            print(f"TF-IDF component values for {ngram} are all 0 in explainer training set.")


def check_explainer_example():
    results_path = "results/top_10000_ngrams"
    llms = ["gpt3.04", "gpt3.041", "gpt3.042", "gpt3.043", "gpt3.5", "gpt4_1106_cot", "gpt4_1106", "llama007"]

    plot_type = "violin"
    num_background_points = 200
    seed = 10
    max_ngram_display = 20

    explainer = Explainer(results_path=results_path, llms=llms, )

    # # Stratified plots
    # strats = ["query_type", "sensical", "phenomenon", ]  # ["query_type", "sensical", "phenomenon", "rung",]
    # explainer.produce_stratified_shap_plots_with_multiple_llms(strats=strats,
    #                                                            plot_type=plot_type,
    #                                                            num_background_points=num_background_points,
    #                                                            seed=seed,
    #                                                            max_ngram_display=max_ngram_display)

    # General, unstratified plots
    explainer.produce_unstratified_shap_plots(plot_type=plot_type,
                                              num_background_points=num_background_points,
                                              seed=seed,
                                              max_ngram_display=max_ngram_display,
                                              )

    print("Exit ok")


if __name__ == "__main__":
    check_explainer_example()
