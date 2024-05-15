""" This file is made up of code snippets and functions written by various authors (Google or Univ. of Cambridge)."""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.calibration import calibration_curve

from src.utils.utils import brier_decomposition
from src.utils.utils import try_mkdir


class Trainer:
    def __init__(self,
                 llms,
                 results_path,
                 num_classes=2,
                 learning_rate=3 * 1e-4,
                 epochs=100,
                 batch_size=128,
                 layers=2,
                 units=64,
                 dropout_rate=0.2):
        self.llms = llms
        self.results_path = results_path
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = layers
        self.units = units
        self.dropout_rate = dropout_rate

    def train_ngram_model(self,
                          x_train_dict,
                          x_val_dict,
                          x_test_dict,
                          train_labels_dict,
                          val_labels_dict, ):
        """Trains n-gram model on the given dataset.

        # Arguments
            data: tuples of training and test texts and labels.
            learning_rate: float, learning rate for training model.
            epochs: int, number of epochs.
            batch_size: int, number of samples per batch.
            layers: int, number of `Dense` layers in the model.
            units: int, output dimension of Dense layers in the model.
            dropout_rate: float: percentage of input to drop at Dropout layers.

        # Raises
            ValueError: If validation data has label values which were not seen
                in the training data.
        """
        # Initialise return variables
        history_acc_dict = {}
        history_val_acc_dict = {}
        y_pred_dict = {}

        for llm in self.llms:
            x_train = x_train_dict[llm]
            x_val = x_val_dict[llm]
            x_test = x_test_dict[llm]
            train_labels = train_labels_dict[llm]
            val_labels = val_labels_dict[llm]

            # Create model instance.
            model = self._create_mlp_model(layers=self.layers,
                                           units=self.units,
                                           dropout_rate=self.dropout_rate,
                                           input_shape=x_train.shape[1:],
                                           num_classes=self.num_classes)

            # Compile model with learning parameters.
            if self.num_classes == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'sparse_categorical_crossentropy'
            optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.learning_rate)  # Use legacy.Adam if you are on M1/M2 mac
            model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

            # Create callback for early stopping on validation loss. If the loss does
            # not decrease in two consecutive tries, stop training.
            callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor='val_acc', patience=10)]  # val_loss

            # Train and validate model.
            history = model.fit(
                x_train,
                train_labels,
                epochs=self.epochs,
                # callbacks=callbacks, # UNCOMMENT TO GET EARLY STOPPING
                validation_data=(x_val, val_labels),
                verbose=2,  # Logs once per epoch.
                batch_size=self.batch_size
            )
            # print(4)
            # Print results.
            history = history.history
            # print('Training accuracy: {acc}, loss: {loss}'.format( #Validation accuracy; loss
            #        acc=history['acc'][-1], loss=history['loss'][-1])) #val_acc, val_loss

            # Save model.
            model.save(f'{self.results_path}/{llm}/model.h5')
            y_pred = model.predict(x_test).ravel()

            # Add the llm-specific arrays to the return dicts
            history_acc_dict[llm] = history['acc']
            history_val_acc_dict[llm] = history['val_acc']
            y_pred_dict[llm] = y_pred

        return history_acc_dict, history_val_acc_dict, y_pred_dict

    def produce_all_training_plots(self,
                                   test_dict,
                                   history_acc_dict,
                                   history_val_acc_dict,
                                   y_pred_dict, ):
        # Create the root directory for the plots
        plot_root_path = f"{self.results_path}/training_plots"
        try_mkdir(plot_root_path)

        res = pd.DataFrame()
        probabilities = {}
        probabilities_e = {}
        training_output = {}
        epsilon = 0.0001

        # Calculate all the summary statistics
        for llm in self.llms:
            print(llm)
            acc, val_acc, y_pred = history_acc_dict[llm], history_val_acc_dict[llm], y_pred_dict[llm]
            y_pred_e = y_pred + epsilon
            probabilities[llm] = y_pred
            probabilities_e[llm] = y_pred_e
            training_output[llm] = pd.DataFrame({'training_accuracy': acc, 'validation_accuracy': val_acc})
            method_name = 'n_gram'
            BrierScore, Calibration, Refinement = brier_decomposition(probabilities[llm], test_dict[llm]['success'])
            # BrierScore_e, Calibration_e, Refinement_e = brierDecomp(probabilities[llm], test_dict[llm]['success'])
            roc_auc = roc_auc_score(test_dict[llm]['success'], probabilities[llm])
            # brier_score_loss_sklearn = brier_score_loss(test_dict[llm]['success'], probabilities[llm])
            prec = precision_score(test_dict[llm]['success'], [1 if p > 0.5 else 0 for p in probabilities[llm]])
            recall = recall_score(test_dict[llm]['success'], [1 if p > 0.5 else 0 for p in probabilities[llm]])
            f1 = f1_score(test_dict[llm]['success'], [1 if p > 0.5 else 0 for p in probabilities[llm]])
            # compute accuracy by thresholding at 0.5
            y_pred_binary = probabilities[llm] > 0.5
            accuracy = np.mean(y_pred_binary == test_dict[llm]['success'])
            res = pd.concat([res, pd.DataFrame(
                {"predictive_method": method_name, "llm": llm, "BrierScore": BrierScore,
                 "Calibration": Calibration, "Refinement": Refinement, "AUROC": roc_auc,
                 'precision': prec, 'recall': recall, 'f1 score': f1, 'accuracy': accuracy},
                index=[0])])

        res.sort_values(by="BrierScore")

        self._plot_learning_curves(plot_root_path, training_output)
        self._plot_histogram_of_prediction_probabilities(plot_root_path, probabilities, res)
        self._plot_calibration_curves(plot_root_path, probabilities, test_dict)
        self._plot_reliability_diagrams(plot_root_path, probabilities, test_dict)

    def _get_last_layer_units_and_activation(self, ):
        """Gets the # units and activation function for the last network layer.

        # Arguments
            num_classes: int, number of classes.

        # Returns
            units, activation values.
        """
        if self.num_classes == 2:
            activation = 'sigmoid'
            units = 1
        else:
            activation = 'softmax'
            units = self.num_classes
        return units, activation

    def _create_mlp_model(self, layers, units, dropout_rate, input_shape, num_classes):
        """Creates an instance of a multi-layer perceptron model.

        # Arguments
            layers: int, number of `Dense` layers in the model.
            units: int, output dimension of the layers.
            dropout_rate: float, percentage of input to drop at Dropout layers.
            input_shape: tuple, shape of input to the model.
            num_classes: int, number of output classes.

        # Returns
            An MLP model instance.
        """
        op_units, op_activation = self._get_last_layer_units_and_activation()
        model = models.Sequential()
        model.add(Dropout(rate=dropout_rate, input_shape=input_shape))
        for _ in range(layers - 1):
            model.add(Dense(units=units, activation='relu'))
            model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=op_units, activation=op_activation))
        return model

    def _plot_learning_curves(self, plot_root_path, training_output):
        # Determine the layout of the subplots
        n_llms = len(self.llms)
        ncols = 4
        nrows = n_llms // ncols + (n_llms % ncols > 0)

        plt.figure(figsize=(ncols * 10, nrows * 8))

        for index, llm in enumerate(training_output):
            ax = plt.subplot(nrows, ncols, index + 1)

            # Create a DataFrame for the current llm
            df_loss = pd.DataFrame({
                'Epoch': range(len(training_output[llm]['training_accuracy'])),
                'Training accuracy': training_output[llm]['training_accuracy'],
                'Validation accuracy': training_output[llm]['validation_accuracy']
            })

            # Melt the DataFrame to have appropriate format for seaborn lineplot
            df_loss_melted = df_loss.melt(id_vars=['Epoch'], var_name='Type', value_name='Accuracy')

            # Plot the training and validation loss
            sns.lineplot(data=df_loss_melted, x='Epoch', y='Accuracy', hue='Type', marker='o', ax=ax)

            ax.set_title(f'Accuracy for {llm}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{plot_root_path}/learning_curves.png")

    def _plot_histogram_of_prediction_probabilities(self, plot_root_path, probabilities, res):
        # Plot histogram of the prediction probabilities for each llm
        # Determine the layout of the subplots
        n_llms = len(self.llms)
        ncols = 4
        nrows = n_llms // ncols + (n_llms % ncols > 0)

        plt.figure(figsize=(ncols * 10, nrows * 8))

        for index, llm in enumerate(self.llms):
            plt.subplot(nrows, ncols, index + 1)

            sns.histplot(probabilities[llm], bins=10, kde=False)
            plt.title(f"Prediction probabilities for {llm}")
            plt.xlim(0, 1)
            acc = res['accuracy'][res['llm'] == llm]
            # print(acc[0])
            plt.axvline(x=acc[0], color='r', linewidth=3, linestyle='--', label='accuracy')
            plt.axvline(x=probabilities[llm].mean(), color='g', linewidth=3, linestyle='-', label='average confidence')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{plot_root_path}/histogram_prediction_probabilities.png")

    def _plot_calibration_curves(self, plot_root_path, probabilities, test_dict):
        plt.figure(figsize=(12, 8))
        n_bins = 10

        # Plot calibration curves for all models on the same axes
        for llm in self.llms:
            prob_true, prob_pred = calibration_curve(test_dict[llm]['success'], probabilities[llm], n_bins=n_bins,
                                                     strategy='uniform')
            sns.lineplot(x=prob_pred, y=prob_true, marker='o', label=f'{llm}')

        # Adding the reference line for perfect calibration
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title("Calibration Curves for Different Models")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.legend(title='Model')
        plt.savefig(f"{plot_root_path}/calibration_curves.png")

    def _plot_reliability_diagrams(self, plot_root_path, probabilities, test_dict):
        # Plot the reliability diagrams
        M = 10
        n_llms = len(self.llms)
        ncols = 4  # You can adjust the number of columns as needed
        nrows = n_llms // ncols + (n_llms % ncols > 0)

        plt.figure(figsize=(ncols * 10, nrows * 6))

        for index, llm in enumerate(self.llms):
            plt.subplot(nrows, ncols, index + 1)

            df_temp = pd.DataFrame({
                'predictions': probabilities[llm],
                'true_label': test_dict[llm]['success']
            })

            bin_data = []

            for m in range(1, M + 1):
                bin = df_temp[(df_temp['predictions'] > (m - 1) / M) & (df_temp['predictions'] <= m / M)]
                accuracy = (bin['true_label']).mean()
                confidence = bin['predictions'].mean()
                error = (len(bin) / len(df_temp)) * (abs(accuracy - confidence))
                # print(len(bin),len(df_temp))

                bin_data.append({'confidence': confidence, 'accuracy': accuracy, 'error': error})

            # Convert to DataFrame for Seaborn
            bin_df = pd.DataFrame(bin_data)
            sum_error = bin_df['error'].sum()
            sns.lineplot(x='confidence', y='accuracy', data=bin_df, marker='o', label=f'{llm}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.title(f"Reliability Diagram for {llm}")
            plt.xlabel('Mean Confidence')
            plt.ylabel('Accuracy')

            plt.annotate(f'Expected Callibration Error = {sum_error:.2f}', xy=(0.5, 0.1), xycoords='axes fraction',
                         ha='center', va='bottom', color='red')

        plt.tight_layout()
        plt.savefig(f"{plot_root_path}/reliability_diagrams.png")

    def _plot_histogram_of_prediction_probability_with_epsilon_offset(self, plot_root_path, probabilities_e):
        # Plot prediction probability histogram distribution when there is an epsilon offset
        # Determine the layout of the subplots
        n_llms = len(self.llms)
        ncols = 4
        nrows = n_llms // ncols + (n_llms % ncols > 0)

        plt.figure(figsize=(ncols * 10, nrows * 8))

        for index, llm in enumerate(self.llms):
            plt.subplot(nrows, ncols, index + 1)

            sns.histplot(probabilities_e[llm], bins=10, kde=False)
            plt.title(f"Prediction probabilities for {llm} with epsilon offset")
            plt.xlim(0, 1)

        plt.tight_layout()
        plt.savefig(f"{plot_root_path}/histogram_prediction_probability_epsilon_offset.png")


def check_trainer_example():
    # Preprocess the data
    from preprocessor import Preprocessor

    kwargs_preprocessor = {
        "llms": ["gpt3.04", "gpt3.5", "gpt3.041", "gpt3.042", "gpt3.043", "gpt4_1106_cot", "gpt4_1106", "llama007"],
        "path_to_dataset_outputs": "datasets/cladder/outputs",
        "ngram_range": (1, 2),
        "top_k_ngrams": 20_000,
        "token_mode": "word",
        "min_document_frequency": 2,
        "num_classes": 2,
        "split_ratio": 0.2,
        "results_path": "results",
    }
    preprocessor = Preprocessor(**kwargs_preprocessor)
    preprocessor.preprocess_dataset()

    # Train the MLPs
    kwargs_trainer = {
        "llms": ["gpt3.04", "gpt3.5", "gpt3.041", "gpt3.042", "gpt3.043", "gpt4_1106_cot", "gpt4_1106", "llama007"],
        "num_classes": 2,
        "results_path": "results"
    }
    kwargs_train_method = {
        "x_train_dict": preprocessor.x_train_dict,
        "x_val_dict": preprocessor.x_val_dict,
        "x_test_dict": preprocessor.x_test_dict,
        "train_labels_dict": preprocessor.train_labels_dict,
        "val_labels_dict": preprocessor.val_labels_dict,
    }

    trainer = Trainer(**kwargs_trainer)

    trainer.train_ngram_model(**kwargs_train_method)

    print("Exit ok")


if __name__ == "__main__":
    check_trainer_example()
