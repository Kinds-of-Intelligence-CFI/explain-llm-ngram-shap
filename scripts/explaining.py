import numpy as np
import pickle
import shap
import tensorflow as tf
from tensorflow.keras.saving import load_model

with open('results/x_train_dict.pickle', 'rb') as handle:
    x_train_dict = pickle.load(handle)

with open('results/x_val_dict.pickle', 'rb') as handle:
    x_val_dict = pickle.load(handle)

with open('results/x_test_dict.pickle', 'rb') as handle:
    x_test_dict = pickle.load(handle)

with open('results/ngrams_dict.pickle', 'rb') as handle:
    ngrams_dict = pickle.load(handle)

results_path = "results"

llms = ["gpt3.04",
        "gpt3.041",
        "gpt3.042",
        "gpt3.043",
        "gpt3.5",
        "gpt4_1106",
        "llama007"]

for llm in llms:
    # Load the model
    model = load_model(f"{results_path}/{llm}_model.h5")

    # Extract the llm-specific data
    x_train = x_train_dict[llm]
    x_test = x_test_dict[llm]
    ngrams = ngrams_dict[llm]

    np.random.seed(0)
    num_rand_idx = 100
    random_indices = np.random.choice(x_train.shape[0], size=num_rand_idx, replace=False)
    random_training_samples = x_train[random_indices, :]

    explainer = shap.DeepExplainer(model, data=random_training_samples)
    shap_values = explainer.shap_values(x_test)
    base_value = float(explainer.expected_value[0])
    explanation = shap.Explanation(values=shap_values[:, :, 0],
                                   base_values=base_value,
                                   data=x_test,
                                   feature_names=ngrams)

    shap.summary_plot(shap_values[:, :, 0], x_test, feature_names=ngrams, plot_type="bar")


