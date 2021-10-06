import os
import pytest
import numpy as np
import pandas as pd
from ndg_tools import language as lang
from sklearn.datasets import fetch_20newsgroups

DATA_DIR = "test_data"


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
    strings = np.array(fetch_20newsgroups(data_home=DATA_DIR)["data"][:100], dtype=str)
    funcs_to_try = (
        lambda x: "",
        _scramble_chars,
    )

    def test_string_input(self):
        """
        Test that it can process a single string.
        """
        for func in self.funcs_to_try:
            ref_string = func(self.strings[0])
            tar_string = lang.process_strings(self.strings[0], func)
            assert tar_string == ref_string

    def test_list_input(self):
        """
        Test that it can process a list of strings.
        """
        for func in self.funcs_to_try:
            ref_strings = [func(x) for x in self.strings]
            tar_strings = lang.process_strings(list(self.strings), func)
            assert tar_strings is not ref_strings
            assert tar_strings == ref_strings

    def test_set_input(self):
        """
        Test that it can process a set of strings.
        """
        for func in self.funcs_to_try:
            ref_strings = {func(x) for x in self.strings}
            tar_strings = lang.process_strings(set(self.strings), func)
            assert tar_strings is not ref_strings
            assert tar_strings == ref_strings

    def test_ndarray_input(self):
        """
        Test that it can process an ndarray of strings.
        """
        for func in self.funcs_to_try:
            ref_strings = np.array([func(x) for x in self.strings], dtype=str).reshape(
                10, 10
            )
            tar_strings = lang.process_strings(self.strings.reshape(10, 10), func)
            assert tar_strings is not ref_strings
            assert np.array_equal(tar_strings, ref_strings)

    def test_series_input(self):
        """
        Test that it can process a Series of strings.
        """
        for func in self.funcs_to_try:
            ref_strings = pd.Series(self.strings).map(func)
            tar_strings = lang.process_strings(pd.Series(self.strings), func)
            assert tar_strings is not ref_strings
            assert tar_strings.equals(ref_strings)

    def test_dataframe_input(self):
        """
        Test that it can process a DataFrame of strings.
        """
        reshaped = pd.DataFrame(self.strings.reshape(10, 10))
        for func in self.funcs_to_try:
            ref_strings = reshaped.applymap(func)
            tar_strings = lang.process_strings(reshaped, func)
            assert tar_strings is not ref_strings
            assert tar_strings.equals(ref_strings)

    def test_multiprocessing(self):
        """Tests that multiprocessing does not raise errors."""
        if os.cpu_count() == 1:
            pytest.skip("Only 1 CPU found.")
        for func in self.funcs_to_try:
            lang.process_strings(self.strings, func, n_jobs=-1)


class TestProcessTokens:
    tokdocs = [x.split() for x in fetch_20newsgroups(data_home=DATA_DIR)["data"][:100]]
    funcs_to_try = (lambda x: [], _scramble_tokens)

    def test_flat_list_input(self):
        """
        Test that it can process a list of strings.
        """
        for func in self.funcs_to_try:
            ref_tokens = func(self.tokdocs[0])
            tar_tokens = lang.process_tokens(self.tokdocs[0], func)
            assert tar_tokens is not ref_tokens
            assert tar_tokens == ref_tokens

    def test_nested_list_input(self):
        """
        Test that it can process a list of lists.
        """
        for func in self.funcs_to_try:
            ref_tokdocs = [func(x) for x in self.tokdocs]
            tar_tokdocs = lang.process_tokens(self.tokdocs, func)
            assert tar_tokdocs is not ref_tokdocs
            assert tar_tokdocs == ref_tokdocs

    def test_flat_ndarray_input(self):
        """
        Test that it can process a 1darray of strings.
        """
        tokens = np.array(self.tokdocs[0], dtype=str)

        for func in self.funcs_to_try:
            ref_tokens = np.array(func(tokens), dtype=str)
            tar_tokens = lang.process_tokens(tokens, func)
            assert tar_tokens is not ref_tokens
            assert np.array_equal(tar_tokens, ref_tokens)

    def test_nested_ndarray_input(self):
        """
        Test that it can process a 1darray of lists.
        """
        tokdocs = np.array(self.tokdocs, dtype="O")
        for func in self.funcs_to_try:
            ref_tokdocs = np.array([func(x) for x in tokdocs], dtype="O")
            tar_tokdocs = lang.process_tokens(tokdocs, func)
            assert tar_tokdocs is not ref_tokdocs
            assert np.array_equal(tar_tokdocs, ref_tokdocs)

    def test_flat_series_input(self):
        """
        Test that it can process a series of strings.
        """
        for func in self.funcs_to_try:
            ref_tokens = pd.Series(func(self.tokdocs[0]), dtype="string")
            tar_tokens = lang.process_tokens(pd.Series(self.tokdocs[0]), func)
            assert tar_tokens is not ref_tokens
            assert tar_tokens.equals(ref_tokens)

    def test_nested_series_input(self):
        """
        Test that it can process a series of lists.
        """
        for func in self.funcs_to_try:
            ref_tokdocs = pd.Series(self.tokdocs).map(func)
            tar_tokdocs = lang.process_tokens(pd.Series(self.tokdocs), func)
            assert tar_tokdocs is not ref_tokdocs
            assert tar_tokdocs.equals(ref_tokdocs)

    def test_multiprocessing(self):
        """Tests that multiprocessing does not raise errors."""
        if os.cpu_count() == 1:
            pytest.skip("Only 1 CPU found.")
        for func in self.funcs_to_try:
            lang.process_tokens(self.tokdocs, func, n_jobs=-1)
