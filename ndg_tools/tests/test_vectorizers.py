import os

import nltk
import numpy as np
import pandas as pd
from ndg_tools.sklearn.vectorizers import FreqVectorizer, VaderVectorizer
from ndg_tools.tests import TWEETS_PATH
from sklearn import clone
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


class TestVaderVectorizer:
    data = pd.read_parquet(TWEETS_PATH)
    X = data["text"].copy()
    y = data["polarity"].copy()
    classifier = DecisionTreeClassifier(random_state=35)

    def test_defaults(self):
        pipe = make_pipeline(VaderVectorizer(), clone(self.classifier))
        score = cross_val_score(pipe, self.X, self.y).mean()
        assert 0.5 < score < 1

    def test_grid_search(self):
        pipe = make_pipeline(VaderVectorizer(), clone(self.classifier))
        param_grid = {
            "round_scores": [True, False],
            "preprocessor": [None, lambda x: x.lower(), lambda x: x.upper()],
            "decode_html_entities": [True, False],
            "sparse": [True, False],
            "dtype": [np.float64, np.float32],
        }
        param_grid = {f"vadervectorizer__{k}": v for k, v in param_grid.items()}
        search = HalvingGridSearchCV(
            pipe,
            param_grid,
            n_jobs=-1,
            cv=3,
            error_score="raise",
            random_state=78,
        )
        search.fit(self.X, self.y)
        assert 0.5 < search.best_score_ < 1


class TestFreqVectorizer:
    data = pd.read_parquet(TWEETS_PATH)
    X = data["text"].copy()
    y = data["polarity"].copy()
    classifier = MultinomialNB()

    def test_defaults(self):
        pipe = make_pipeline(FreqVectorizer(), clone(self.classifier))
        score = cross_val_score(pipe, self.X, self.y).mean()
        assert 0.5 < score < 1

    def build_search_estimator(self, param_grid, cv=3, n_jobs=-1, random_state=30):
        pipe = make_pipeline(FreqVectorizer(), clone(self.classifier))
        param_grid = {f"freqvectorizer__{k}": v for k, v in param_grid.items()}
        search = HalvingGridSearchCV(
            pipe,
            param_grid,
            n_jobs=n_jobs,
            cv=cv,
            error_score="raise",
            random_state=random_state,
        )
        return search

    def test_text_filters(self):
        param_grid = {
            "lowercase": [True, False],
            "strip_accents": [None, "ascii", "unicode"],
            "decode_html_entities": [True, False],
            "strip_punct": [True, False, ".?!"],
            "strip_numeric": [True, False],
            "strip_twitter_handles": [True, False],
            "limit_repeats": [True, False],
            "strip_non_word": [True, False],
        }
        search = self.build_search_estimator(param_grid)
        search.fit(self.X, self.y)
        assert 0.5 < search.best_score_ < 1

    def test_token_filters(self):
        param_grid = {
            "tokenizer": [nltk.casual_tokenize],
            "uniq_char_thresh": [None, 0.375],
            "mark_negation": [True, False],
            "stemmer": [None, "porter", "wordnet"],
            "stop_words": [
                None,
                "nltk_english",
                "sklearn_english",
                {"the", "and", "or"},
            ],
            "ngram_range": [(1, 1), (1, 2)],
        }
        search = self.build_search_estimator(param_grid)
        search.fit(self.X, self.y)
        assert 0.5 < search.best_score_ < 1

    def test_mark_negation(self):
        freq_vec = FreqVectorizer(mark_negation=True, tokenizer=nltk.casual_tokenize)
        analyze = freq_vec.build_analyzer()
        vocab = {w for d in self.X for w in analyze(d)}
        tagged = {w for w in vocab if w.endswith("_neg")}
        assert 600 < len(tagged) < 800

    def test_feature_selection(self):
        param_grid = {
            "max_features": [100, 1000, 10000],
            "min_df": [1, 2, 3],
            "max_df": [0.25, 0.5, 0.75, 1.0],
        }
        search = self.build_search_estimator(param_grid)
        search.fit(self.X, self.y)

    def test_tfidf_params(self):
        param_grid = {
            "use_idf": [True, False],
            "binary": [True, False],
            "sublinear_tf": [True, False],
            "smooth_idf": [True, False],
            "norm": [None, "l1", "l2"],
            "dtype": [np.float64, np.float32],
        }
        search = self.build_search_estimator(param_grid)
        search.fit(self.X, self.y)
        assert 0.5 < search.best_score_ < 1
