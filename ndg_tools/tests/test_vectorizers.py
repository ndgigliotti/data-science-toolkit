import os

import numpy as np
import pandas as pd
from ndg_tools.sklearn.vectorizers import FreqVectorizer, VaderVectorizer
from ndg_tools.tests import DATA_DIR
from sklearn import clone
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


class TestVaderVectorizer:
    data = pd.read_parquet(os.path.join(DATA_DIR, "tweets_sentiment.parquet"))
    X = data["text"].copy()
    y = data["polarity"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=24, stratify=y
    )
    classifier = DecisionTreeClassifier(random_state=35)

    def test_defaults(self):
        pipe = make_pipeline(VaderVectorizer(), clone(self.classifier))
        pipe.fit(self.X_train, self.y_train)
        score = accuracy_score(self.y_test, pipe.predict(self.X_test))
        assert score == 0.624

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


class TestFreqVectorizer:
    data = pd.read_parquet(os.path.join(DATA_DIR, "tweets_sentiment.parquet"))
    X = data["text"].copy()
    y = data["polarity"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=24, stratify=y
    )
    classifier = DecisionTreeClassifier(random_state=35)

    def test_defaults(self):
        pipe = make_pipeline(FreqVectorizer(), clone(self.classifier))
        pipe.fit(self.X_train, self.y_train)
        score = accuracy_score(self.y_test, pipe.predict(self.X_test))
        assert score == 0.62

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

    def test_token_filters(self):
        param_grid = {
            "uniq_char_thresh": [None, 0.375],
            "stemmer": [None, "porter", "wordnet"],
            "stop_words": [
                None,
                "nltk_english",
                "gensim_english",
                "skl_english",
                {"the", "and", "or"},
            ],
            "ngram_range": [(1, 1), (1, 2)],
        }
        search = self.build_search_estimator(param_grid)
        search.fit(self.X, self.y)

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
