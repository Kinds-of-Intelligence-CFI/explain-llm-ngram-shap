"""
This file is made up of code written by different authors (both internal and external to Univ. of Cambridge).
In the absence of specification, the code was produced by Matteo G Mecattaf.
"""

import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


class Preprocessor:
    def __init__(self, llms, path_to_dataset_outputs, ngram_range, top_k_ngrams, token_mode, min_document_frequency,
                 split_ratio=0.2):
        self.llms = llms
        self.path_to_results = path_to_dataset_outputs
        self.ngram_range = ngram_range
        self.top_k_ngrams = top_k_ngrams
        self.token_mode = token_mode
        self.min_document_frequency = min_document_frequency
        self.split_ratio = split_ratio

    def preprocess_dataset(self, ):
        results_dict = self._load_dataset_results()
        train_dict, val_dict, test_dict = self._split_data(results_dict)
        x_train_dict, x_val_dict, x_test_dict, ngrams_dict = self._vectorise_prompts(train_dict, val_dict, test_dict)
        return x_train_dict, x_val_dict, x_test_dict, ngrams_dict

    # Authored by Marko Tesic, revised by Matteo G Mecattaf
    def _load_dataset_results(self, ):
        results_dict = {}
        for llm in self.llms:
            results_dict[llm] = pd.read_csv(f"{self.path_to_results}/{llm}.csv")
            results_dict[llm]['success'] = (
                    results_dict[llm]['truth_norm'] == results_dict[llm]['pred_norm']).astype(int)
        return results_dict

    # Authored by Marko Tesic, revised by Matteo G Mecattaf
    def _split_data(self, results_dict, test_size=0.2):
        # Dictionaries to hold the training and test data
        train_dict = {}
        val_dict = {}
        test_dict = {}

        for llm, df in results_dict.items():
            train_data, temp_data = train_test_split(df, test_size=test_size, random_state=42)
            val_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
            train_dict[llm] = train_data
            val_dict[llm] = val_data
            test_dict[llm] = test_data

        return train_dict, val_dict, test_dict

    # Authored by Google, revised by Matteo G Mecattaf
    def _vectorise_prompts(self, train_dict, val_dict, test_dict):
        # Initialise return variables
        x_train_dict = {}
        x_val_dict = {}
        x_test_dict = {}
        ngrams_dict = {}

        for llm in tqdm(self.llms):
            train_texts = train_dict[llm]['prompt']
            val_texts = val_dict[llm]['prompt']
            test_texts = test_dict[llm]['prompt']
            train_labels = train_dict[llm]['success']

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

            x_train_dict[llm] = x_train
            x_val_dict[llm] = x_val
            x_test_dict[llm] = x_test
            ngrams_dict[llm] = n_grams

        return x_train_dict, x_val_dict, x_test_dict, ngrams_dict


if __name__ == "__main__":
    kwargs = {
        "llms": ["gpt3.04", "gpt3.5", "gpt3.041", "gpt3.042", "gpt3.043", "gpt4_1106_cot", "gpt4_1106", "llama007"],
        "path_to_dataset_outputs": "datasets/cladder/outputs",
        "ngram_range": (1, 2),
        "top_k_ngrams": 20_000,
        "token_mode": "word",
        "min_document_frequency": 2,
        "split_ratio": 0.2,
    }

    preprocessor = Preprocessor(**kwargs)
    x_train_dict, x_val_dict, x_test_dict, ngram_dict = preprocessor.preprocess_dataset()

    print("Exit ok")
