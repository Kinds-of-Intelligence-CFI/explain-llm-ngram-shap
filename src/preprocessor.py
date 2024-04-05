"""
This file is made up of code written by different authors (both internal and external to Univ. of Cambridge).
In the absence of specification, the code was produced by Matteo G Mecattaf.
"""

import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


class Preprocessor:
    def __init__(self,
                 llms,
                 dataset_path,
                 ngram_range,
                 top_k_ngrams,
                 token_mode,
                 min_document_frequency,
                 num_classes,
                 results_path="results",
                 split_ratio=0.2, ):
        self.llms = llms
        self.path_to_results = dataset_path
        self.ngram_range = tuple(ngram_range)
        self.top_k_ngrams = top_k_ngrams
        self.token_mode = token_mode
        self.min_document_frequency = min_document_frequency
        self.num_classes = num_classes
        self.split_ratio = split_ratio
        self.results_path = results_path

        self.train_dict = None
        self.val_dict = None
        self.test_dict = None

        self.train_labels_dict = None
        self.val_labels_dict = None
        self.test_labels_dict = None

        self.x_train_dict = None
        self.x_train_dict = None
        self.x_train_dict = None
        self.ngrams_dict = None

    def preprocess_dataset(self, ):
        results_dict = self._load_dataset_results()
        self._split_data(results_dict)
        self._extract_labels_dicts()
        self._vectorise_prompts(self.train_dict, self.val_dict, self.test_dict)

    # Authored by Marko Tesic, revised by Matteo G Mecattaf
    def _load_dataset_results(self,):
        if ".pkl" in self.path_to_results or ".pickle" in self.path_to_results:
            # If path to dataset is given as a pickle file, assume it is already in the results_dict format
            with open(self.path_to_results, "rb") as file:
                results_dict = pickle.load(file)

        else:
            # If path to dataset is a directory, assume that it contains csv files under each LLM's name
            results_dict = {}
            for llm in self.llms:
                results_dict[llm] = pd.read_csv(f"{self.path_to_results}/{llm}.csv")
                results_dict[llm]['success'] = (
                        results_dict[llm]['truth_norm'] == results_dict[llm]['pred_norm']).astype(int)

        return results_dict

    def _extract_labels_dicts(self, ):
        self.train_labels_dict = {llm: self.train_dict[llm]["success"] for llm in self.llms}
        self.val_labels_dict = {llm: self.val_dict[llm]["success"] for llm in self.llms}
        self.test_labels_dict = {llm: self.test_dict[llm]["success"] for llm in self.llms}

    # Authored by Marko Tesic, revised by Matteo G Mecattaf
    def _split_data(self, results_dict, test_size=0.2):
        # Dictionaries to hold the training and test data
        self.train_dict = {}
        self.val_dict = {}
        self.test_dict = {}

        for llm, df in results_dict.items():
            train_data, temp_data = train_test_split(df, test_size=test_size, random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
            self.train_dict[llm] = train_data
            self.val_dict[llm] = val_data
            self.test_dict[llm] = test_data

        with open(f'{self.results_path}/train_dict.pickle', 'wb') as handle:
            pickle.dump(self.train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{self.results_path}/val_dict.pickle', 'wb') as handle:
            pickle.dump(self.val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{self.results_path}/test_dict.pickle', 'wb') as handle:
            pickle.dump(self.test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Authored by Google, revised by Matteo G Mecattaf
    def _vectorise_prompts(self, train_dict, val_dict, test_dict):
        # Initialise return variables
        self.x_train_dict = {}
        self.x_val_dict = {}
        self.x_test_dict = {}
        self.ngrams_dict = {}

        for llm in self.llms:
            train_texts = train_dict[llm]['prompt']
            val_texts = val_dict[llm]['prompt']
            test_texts = test_dict[llm]['prompt']

            train_labels = train_dict[llm]['success']
            val_labels = val_dict[llm]["success"]

            unexpected_labels = [v for v in val_labels if v not in range(self.num_classes)]
            if len(unexpected_labels):
                raise ValueError('Unexpected label values found in the validation set:'
                                 ' {unexpected_labels}. Please make sure that the '
                                 'labels in the validation set are in the same range '
                                 'as training labels.'.format(
                    unexpected_labels=unexpected_labels))

            # Create keyword arguments to pass to the 'tf-idf' vectoriser.
            kwargs = {
                'ngram_range': self.ngram_range,  # Use 1-grams + 2-grams.
                'dtype': np.float64,
                'strip_accents': 'unicode',
                'decode_error': 'replace',
                'analyzer': self.token_mode,  # Split text into word tokens.
                'min_df': self.min_document_frequency,
            }
            vectoriser = TfidfVectorizer(**kwargs)

            # Learn vocabulary from training texts and vectorize training, val, and test sets
            x_train = vectoriser.fit_transform(train_texts).toarray()
            x_val = vectoriser.transform(val_texts).toarray()
            x_test = vectoriser.transform(test_texts).toarray()

            # Select top 'k' of the vectorized features.
            selector = SelectKBest(f_classif, k=min(self.top_k_ngrams, x_train.shape[1]))
            selector.fit(x_train, train_labels)

            # Save the text n-grams for future analysis
            # Get a Boolean mask of the retained feature (n-gram) indices
            # Only retain the n-grams corresponding to the TOP_K TF-IDF vectors
            n_grams = vectoriser.get_feature_names_out()
            mask = selector.get_support(indices=False)
            n_grams = n_grams[mask]

            x_train = selector.transform(x_train).astype('float64')
            x_val = selector.transform(x_val).astype('float64')
            x_test = selector.transform(x_test).astype('float64')

            self.x_train_dict[llm] = x_train
            self.x_val_dict[llm] = x_val
            self.x_test_dict[llm] = x_test
            self.ngrams_dict[llm] = n_grams

        with open(f'{self.results_path}/x_train_dict.pickle', 'wb') as handle:
            pickle.dump(self.x_train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{self.results_path}/x_val_dict.pickle', 'wb') as handle:
            pickle.dump(self.x_val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{self.results_path}/x_test_dict.pickle', 'wb') as handle:
            pickle.dump(self.x_test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{self.results_path}/ngrams_dict.pickle', 'wb') as handle:
            pickle.dump(self.ngrams_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def check_preprocessor_example():
    kwargs = {
        "llms": ["gpt3.04", "gpt3.5", "gpt3.041", "gpt3.042", "gpt3.043", "gpt4_1106_cot", "gpt4_1106", "llama007"],
        "dataset_path": "datasets/cladder/outputs",
        "ngram_range": (1, 2),
        "top_k_ngrams": 20_000,
        "token_mode": "word",
        "min_document_frequency": 2,
        "num_classes": 2,
        "split_ratio": 0.2,
        "results_path": "results/dummy",
    }

    preprocessor = Preprocessor(**kwargs)
    preprocessor.preprocess_dataset()

    print("Exit ok")


if __name__ == "__main__":
    check_preprocessor_example()
