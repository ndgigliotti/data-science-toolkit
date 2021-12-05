import os
import re
from string import digits, punctuation

import numpy as np
import pandas as pd
import pytest
import itertools
from ndg_tools import language as lang, utils
from ndg_tools.tests import DATA_DIR, TWEETS_PATH


def _scramble_chars(string):
    """Randomly resample characters at a random size (slow)."""
    old_len = len(string)
    rng = np.random.default_rng(old_len)
    new_len = round(old_len * rng.random() * 3)
    char_bank = list(string.lower() + string.upper())
    return "".join(rng.choice(char_bank, size=new_len, replace=True))


def _scramble_tokens(tokens):
    """Randomly scramble tokens and characters."""
    return _scramble_chars(" ".join(tokens)).split()


class TestProcessStrings:
    docs = pd.read_parquet(TWEETS_PATH, columns=["text"]).squeeze()
    funcs_to_try = (
        lambda x: "",
        _scramble_chars,
    )

    def test_string_input(self):
        """
        Test that it can process a single string.
        """
        text = self.docs.sample().squeeze()
        for func in self.funcs_to_try:
            ref_string = func(text)
            tar_string = lang.process_strings(text, func)
            assert tar_string == ref_string

    def test_list_input(self):
        """
        Test that it can process a list of strings.
        """
        docs = self.docs.sample(10).to_list()
        for func in self.funcs_to_try:
            ref_strings = [func(x) for x in docs]
            tar_strings = lang.process_strings(docs, func)
            assert tar_strings is not ref_strings
            assert tar_strings == ref_strings

    def test_set_input(self):
        """
        Test that it can process a set of strings.
        """
        docs = set(self.docs.sample(10))
        for func in self.funcs_to_try:
            ref_strings = {func(x) for x in docs}
            tar_strings = lang.process_strings(docs, func)
            assert tar_strings is not ref_strings
            assert tar_strings == ref_strings

    def test_ndarray_input(self):
        """
        Test that it can process an ndarray of strings.
        """
        docs = self.docs.sample(10).to_numpy(dtype=str).reshape(-1, 2)
        for func in self.funcs_to_try:
            ref_strings = np.array([func(x) for x in docs.flat], dtype=str).reshape(
                -1, 2
            )
            tar_strings = lang.process_strings(docs, func)
            assert tar_strings is not ref_strings
            assert np.array_equal(tar_strings, ref_strings)

    def test_series_input(self):
        """
        Test that it can process a Series of strings.
        """
        docs = self.docs.sample(10)
        for func in self.funcs_to_try:
            ref_strings = docs.map(func)
            tar_strings = lang.process_strings(docs, func)
            assert tar_strings is not ref_strings
            assert tar_strings.equals(ref_strings)

    def test_dataframe_input(self):
        """
        Test that it can process a DataFrame of strings.
        """
        docs = pd.DataFrame(self.docs.sample(10).to_numpy().reshape(-1, 2))
        for func in self.funcs_to_try:
            ref_strings = docs.applymap(func)
            tar_strings = lang.process_strings(docs, func)
            assert tar_strings is not ref_strings
            assert tar_strings.equals(ref_strings)

    def test_multiprocessing(self):
        """Use multiprocessing to see if errors are raised."""
        if os.cpu_count == 1:
            pytest.skip("Found only 1 CPU.")
        else:
            for func in self.funcs_to_try:
                lang.process_strings(self.docs.to_list(), func, n_jobs=-1)


class TestProcessTokens:
    docs = pd.read_parquet(TWEETS_PATH, columns=["text"]).squeeze()
    tokdocs = docs.str.split()
    funcs_to_try = (lambda x: [], _scramble_tokens)

    def test_flat_list_input(self):
        """
        Test that it can process a list of strings.
        """
        tokens = list(self.tokdocs.sample().squeeze())
        for func in self.funcs_to_try:
            ref_tokens = func(tokens)
            tar_tokens = lang.process_tokens(tokens, func)
            assert tar_tokens is not ref_tokens
            assert tar_tokens == ref_tokens

    def test_nested_list_input(self):
        """
        Test that it can process a list of lists.
        """
        tokdocs = self.tokdocs.sample(10).to_list()
        for func in self.funcs_to_try:
            ref_tokdocs = [func(x) for x in tokdocs]
            tar_tokdocs = lang.process_tokens(tokdocs, func)
            assert tar_tokdocs is not ref_tokdocs
            assert tar_tokdocs == ref_tokdocs

    def test_flat_ndarray_input(self):
        """
        Test that it can process a 1darray of strings.
        """
        tokens = np.array(self.tokdocs.sample().squeeze(), dtype=str)
        for func in self.funcs_to_try:
            ref_tokens = np.array(func(tokens), dtype=str)
            tar_tokens = lang.process_tokens(tokens, func)
            assert tar_tokens is not ref_tokens
            assert np.array_equal(tar_tokens, ref_tokens)

    def test_nested_ndarray_input(self):
        """
        Test that it can process a 1darray of lists.
        """
        tokdocs = self.tokdocs.sample(10).to_numpy()
        for func in self.funcs_to_try:
            ref_tokdocs = np.array([func(x) for x in tokdocs], dtype="O")
            tar_tokdocs = lang.process_tokens(tokdocs, func)
            assert tar_tokdocs is not ref_tokdocs
            assert np.array_equal(tar_tokdocs, ref_tokdocs)

    def test_flat_series_input(self):
        """
        Test that it can process a series of strings.
        """
        tokens = list(self.tokdocs.sample().squeeze())
        for func in self.funcs_to_try:
            ref_tokens = pd.Series(func(tokens), dtype="string")
            tar_tokens = lang.process_tokens(pd.Series(tokens), func)
            assert tar_tokens is not ref_tokens
            assert tar_tokens.equals(ref_tokens)

    def test_nested_series_input(self):
        """
        Test that it can process a series of lists.
        """
        tokdocs = self.tokdocs.sample(10)
        for func in self.funcs_to_try:
            ref_tokdocs = tokdocs.map(func)
            tar_tokdocs = lang.process_tokens(tokdocs, func)
            assert tar_tokdocs is not ref_tokdocs
            assert tar_tokdocs.equals(ref_tokdocs)

    def test_multiprocessing(self):
        """Use multiprocessing to see if errors are raised."""
        if os.cpu_count == 1:
            pytest.skip("Found only 1 CPU.")
        else:
            for func in self.funcs_to_try:
                lang.process_tokens(self.tokdocs.to_list(), func, n_jobs=-1)


class TestTextProcessors:
    docs = pd.read_parquet(TWEETS_PATH, columns=["text"]).squeeze()

    def test_strip_punct(self):
        spaces = lang.strip_punct(punctuation)
        empty = lang.strip_punct(punctuation, repl="")
        excl_point = lang.strip_punct(punctuation, repl="", exclude="!")
        assert spaces == " " * len(punctuation)
        assert empty == ""
        assert excl_point == "!"

    def test_decode_html_entities(self):
        entity_table = pd.read_json(os.path.join(DATA_DIR, "html_entities.json"))
        symbols = lang.decode_html_entities(entity_table["name"])
        assert symbols.equals(entity_table["symbol"])

    def test_strip_numeric(self):
        assert lang.strip_numeric(digits) == ""
        clean_docs = lang.strip_numeric(self.docs.sample(100, random_state=63))
        assert not clean_docs.str.contains(r"\d").any()
        assert clean_docs.str.contains(r"\b\w+\b").all()

    def test_strip_end_space(self):
        dirty = " \n\r\f\n\n     blah blah      \t\v\v\f\t\n "
        clean = "blah blah"
        assert lang.strip_end_space(dirty) == clean

    def test_strip_extra_space(self):
        dirty = "\t blah blah   \t  blah\n\n blah \r\fblah\f\t blah\n\v blah  \n\t"
        clean = "blah blah blah blah blah blah blah"
        assert lang.strip_extra_space(dirty) == clean

    def test_strip_non_word(self):
        clean_docs = lang.strip_non_word(self.docs.sample(100, random_state=16))
        assert not clean_docs.str.contains(r"[^\w ]").any()
        assert clean_docs.str.contains(r"\b\w+\b").all()


class TestTokenProcessors:
    docs = pd.read_parquet(TWEETS_PATH, columns=["text"]).squeeze()
    rng = np.random.default_rng(894)

    def test_fetch_stopwords(self):
        alias = ["nltk", "sklearn"]
        atomic = [
            "nltk_english",
            "nltk_spanish",
            "nltk_german",
            "sklearn_english",
        ]
        operators = ["|", "&", "-", "^"]

        # Construct complexity 1 queries
        level_1 = [" ".join(x) for x in utils.cartesian(atomic, operators, atomic)]
        wrapped = [f"({x})" for x in level_1]

        # Construct (sample of) complexity 2 queries
        level_2 = utils.cartesian(wrapped + atomic, operators, wrapped + atomic)
        sample_idx = self.rng.choice(len(level_2), len(level_2) // 100)
        level_2 = [" ".join(x) for x in np.take(level_2, sample_idx, axis=0)]

        # Test queries
        for query in alias + atomic:
            stopwords = lang.fetch_stopwords(query)
            assert stopwords
            assert all([isinstance(x, str) for x in stopwords])
        for query in level_1:
            stopwords = lang.fetch_stopwords(query)
            if "|" in query:
                assert stopwords
            assert all([isinstance(x, str) for x in stopwords])
        for query in level_2:
            stopwords = lang.fetch_stopwords(query)
            assert all([isinstance(x, str) for x in stopwords])


def test_length_dist():
    df = pd.read_parquet(TWEETS_PATH, columns=["text", "handle"])
    fig = lang.length_dist(df, tick_prec=1)
    assert len(fig.get_axes()) == 2

    # Check labels and ticks
    for ax, column in zip(fig.get_axes(), df.columns):
        assert ax.get_xlabel() == "Character Count"
        assert ax.get_ylabel() == "Document Count"
        assert ax.get_title().startswith(f"Length of '{column}'")
        ticks = [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
        assert all([re.fullmatch(r"-?\d+\.\dK?", t) for t in ticks])
