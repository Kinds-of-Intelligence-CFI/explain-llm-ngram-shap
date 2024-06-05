"""This file is made up of code snippets and functions written by various authors (Google or Univ. of Cambridge)."""
import pickle
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


class Preprocessor:
    RANDOM_SEED_SPLITTING = 42
    def __init__(self,
                 llms: List[str],
                 dataset_path: str,
                 ngram_range: List[int],
                 top_k_ngrams: int,
                 token_mode: str,
                 min_document_frequency: int,
                 num_classes: int,
                 results_path: str = "results",
                 split_ratio: float = 0.2,
                 target_variable: str = "success") -> None:
        self.llms = llms
        self.path_to_results = dataset_path
        self.ngram_range = tuple(ngram_range)
        self.top_k_ngrams = top_k_ngrams
        self.token_mode = token_mode
        self.min_document_frequency = min_document_frequency
        self.num_classes = num_classes
        self.split_ratio = split_ratio
        self.results_path = results_path

        POSSIBLE_TARGET_VARIABLES = ["success", "truth_norm"]
        if target_variable not in POSSIBLE_TARGET_VARIABLES:
            raise ValueError(f"Please pass a dataset_fold from the following options: {POSSIBLE_TARGET_VARIABLES}")

        self.target_variable = target_variable

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

    def preprocess_dataset(self) -> None:
        results_dict = self._load_pickled_dataset_results()
        self._split_data_into_train_val_test_sets(results_dict)
        self._extract_labels_dicts()
        self._vectorise_prompts(self.train_dict, self.val_dict, self.test_dict)

    def _load_pickled_dataset_results(self) -> Dict:
        with open(self.path_to_results, "rb") as file:
            results_dict = pickle.load(file)
        return results_dict

    def _load_folder_of_csvs_dataset_results(self) -> Dict:
        # If path to dataset is a directory, assume that it contains csv files under each LLM's name
        results_dict = {}
        for llm in self.llms:
            results_dict[llm] = pd.read_csv(f"{self.path_to_results}/{llm}.csv")
            results_dict[llm]['success'] = (
                    results_dict[llm]['truth_norm'] == results_dict[llm]['pred_norm']).astype(int)
        return results_dict

    def _extract_labels_dicts(self) -> None:
        self.train_labels_dict = {llm: self.train_dict[llm][self.target_variable] for llm in self.llms}
        self.val_labels_dict = {llm: self.val_dict[llm][self.target_variable] for llm in self.llms}
        self.test_labels_dict = {llm: self.test_dict[llm][self.target_variable] for llm in self.llms}

    def _split_data_into_train_val_test_sets(self,
                                             complete_data: Dict,
                                             test_size: float = 0.2) -> None:
        # Dictionaries to hold the training and test data
        self.train_dict = {}
        self.val_dict = {}
        self.test_dict = {}

        for llm, df in complete_data.items():
            train_data, temp_data = train_test_split(df, test_size=test_size, random_state=self.RANDOM_SEED_SPLITTING)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=self.RANDOM_SEED_SPLITTING)
            self.train_dict[llm] = train_data
            self.val_dict[llm] = val_data
            self.test_dict[llm] = test_data

        with open(f'{self.results_path}/train_dict.pickle', 'wb') as handle:
            pickle.dump(self.train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{self.results_path}/val_dict.pickle', 'wb') as handle:
            pickle.dump(self.val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{self.results_path}/test_dict.pickle', 'wb') as handle:
            pickle.dump(self.test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _vectorise_prompts(self,
                           train_dict: Dict,
                           val_dict: Dict,
                           test_dict: Dict) -> None:
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


def check_preprocessor_example() -> None:
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


if __name__ == "__main__":
    check_preprocessor_example()
