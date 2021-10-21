import copy
import string
from functools import partial
from typing import Callable
import warnings
import nltk

import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import mark_negation as nltk_mark_negation
from pandas.core.series import Series
from scipy.sparse import csr_matrix
from ndg_tools import language as lang
from ndg_tools import utils
from ndg_tools._validation import _invalid_value, _validate_raw_docs
from ndg_tools.typing import CallableOnStr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
    _VectorizerMixin,
    strip_accents_ascii,
    strip_accents_unicode,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer
from sklearn.utils import deprecated
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed


class VaderVectorizer(BaseEstimator, TransformerMixin):
    """Extracts VADER polarity scores from short documents.

    Parameters
    ----------

    round_scores : bool, optional
        Round scores to the nearest integer. By default False.
    preprocessor : CallableOnStr, optional
        Callable for preprocessing text before VADER analysis, by default None.
    decode_html_entities: bool, optional
        Decode HTML entities such as '&mdash;' or '&lt;' or '&gt;' into symbols,
        e.g. '—', '<', '>'. True by default.
    n_jobs: int, optional
        Number of processes for computing scores. Defaults to None.
    dtype: dtype, str
        Type of output elements. Defaults to `np.float64`.
    sparse : bool, optional
        Output a sparse matrix, by default False.
    """

    def __init__(
        self,
        round_scores=False,
        preprocessor: CallableOnStr = None,
        decode_html_entities=True,
        n_jobs=None,
        dtype=np.float64,
        sparse=False,
    ):
        nltk.download("vader_lexicon")
        self.round_scores = round_scores
        self.preprocessor = preprocessor
        self.decode_html_entities = decode_html_entities
        self.n_jobs = n_jobs
        self.dtype = dtype
        self.sparse = sparse
        self._vader = SentimentIntensityAnalyzer()

    def _validate_params(self):
        """Validate some parameters."""
        if self.preprocessor is not None:
            if not isinstance(self.preprocessor, Callable):
                raise TypeError(
                    f"Expected `preprocessor` to be callable, got {type(self.preprocessor)}"
                )

    @deprecated("Use `get_feature_names_out` instead.")
    def get_feature_names(self):
        """Return list of feature names."""
        return self.get_feature_names_out()

    def get_feature_names_out(self, input_features=None):
        """Return list of feature names. `input_features` does nothing."""
        feat_names = ["neg", "neu", "pos", "compound"]
        return feat_names

    def fit(self, raw_documents, y=None):
        """Does nothing except validate parameters."""
        self._validate_params()
        _validate_raw_docs(raw_documents)
        return self

    def transform(self, raw_documents):
        """Extracts the polarity scores and applies postprocessing."""
        # Input and param validation
        self._validate_params()
        _validate_raw_docs(raw_documents)
        workers = Parallel(n_jobs=self.n_jobs, prefer="processes")

        # Apply preprocessing
        docs = raw_documents
        if self.decode_html_entities:
            decode = delayed(lang.decode_html_entities)
            docs = workers(decode(x) for x in docs)
        if self.preprocessor is not None:
            docs = [self.preprocessor(x) for x in docs]

        # Perform VADER analysis
        polarity_scores = delayed(self._vader.polarity_scores)
        vecs = pd.DataFrame(workers(polarity_scores(x) for x in docs), dtype=self.dtype)
        vecs = vecs.loc[:, self.get_feature_names_out()].to_numpy()

        # Apply postprocessing and return
        if self.round_scores:
            vecs = vecs.round()
        if self.sparse:
            vecs = csr_matrix(vecs)
        return vecs

    def fit_transform(self, raw_documents, y=None):
        """Extracts the polarity scores and applies postprocessing."""
        return self.transform(raw_documents)


class VectorizerMixin(_VectorizerMixin):
    def _word_ngrams(self, tokens, stop_words=None, sep="_"):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            join = sep.join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(join(original_tokens[i : i + n]))
        return tokens

    def build_preprocessor(self):
        if self.preprocessor is not None:
            return self.preprocessor

        pipe = []

        # Make case insensitive
        if self.lowercase:
            pipe.append(lang.lowercase)

        # Decode HTML entities like '&mdash;' to '—'
        if self.decode_html_entities:
            pipe.append(lang.decode_html_entities)

        # Strip accents
        if not self.strip_accents:
            pass
        elif callable(self.strip_accents):
            pipe.append(self.strip_accents)
        elif self.strip_accents == "ascii":
            pipe.append(strip_accents_ascii)
        elif self.strip_accents == "unicode":
            pipe.append(strip_accents_unicode)
        else:
            _invalid_value("strip_accents", self.strip_accents)

        # Strip HTML tags
        if self.strip_html_tags:
            pipe.append(lang.strip_html_tags)

        # Strip numerals
        if self.strip_numeric:
            pipe.append(lang.strip_numeric)

        # Strip Twitter @handles
        if self.strip_twitter_handles:
            pipe.append(lang.strip_twitter_handles)

        # Strip punctuation
        if self.strip_punct:
            if isinstance(self.strip_punct, str):
                pipe.append(partial(lang.strip_punct, punct=self.strip_punct))
            else:
                pipe.append(lang.strip_punct)

        # Strip all non-word characters (non-alphanumeric)
        if self.strip_non_word:
            pipe.append(lang.strip_non_word)

        if self.limit_repeats:
            pipe.append(lang.limit_repeats)

        # Strip extra whitespaces, tabs, and linebreaks
        if self.strip_extra_space:
            pipe.append(lang.strip_extra_space)

        # Wrap `pipe` into single callable
        return lang.make_preprocessor(pipe)

    def build_analyzer(self):
        """Return the complete text preprocessing pipeline as a callable.

        Handles decoding, character filtration, tokenization, word filtration,
        marking, and n-gram generation. Alternatively, returns a callable
        wrapping the custom analyzer passed via the `analyzer` parameter.

        Returns
        -------
        analyzer: callable
            A function to handle decoding, character filtration, tokenization,
            word filtration, n-gram generation, and marking.
        """
        pipe = [self.decode]

        if callable(self.analyzer):
            pipe.append(self.analyzer)
        elif self.analyzer == "char":
            pipe += [self.build_preprocessor(), self._char_ngrams]
        elif self.analyzer == "char_wb":
            pipe += [self.build_preprocessor(), self._char_wb_ngrams]
        elif self.analyzer == "word":
            preprocessor = self.build_preprocessor()
            tokenizer = self.build_tokenizer()
            pipe += [preprocessor, tokenizer]

            if self.uniq_char_thresh is not None:
                pipe.append(
                    partial(lang.uniq_char_thresh, thresh=self.uniq_char_thresh)
                )

            # Stem or lemmatize
            if callable(self.stemmer):
                pipe.append(self.stemmer)
            elif self.stemmer == "porter":
                pipe.append(lang.porter_stem)
            elif self.stemmer == "wordnet":
                pipe.append(lang.wordnet_lemmatize)

            if self.mark_negation:
                pipe.append(nltk_mark_negation)
                if self.lowercase:
                    # Make tags lowercase to avoid warning
                    pipe.append(lang.lowercase)

            # Remove stopwords
            if self.stop_words is not None:
                stop_words = self.get_stop_words()
                self._check_stop_words_consistency(stop_words, preprocessor, tokenizer)
                pipe.append(partial(lang.remove_stopwords, stopwords=stop_words))

            # Generate n-grams
            pipe.append(self._word_ngrams)

        else:
            raise _invalid_value("analyzer", self.analyzer, ("word", "char", "char_wb"))

        # Wrap `pipe` into single callable
        return lang.make_preprocessor(pipe)

    def get_stop_words(self):
        """Build or fetch the effective stop words set.

        Returns
        -------
        stop_words: frozenset or None
                A set of stop words.
        """
        # Exit if None
        if self.stop_words is None:
            return None
        # Process string input
        elif isinstance(self.stop_words, str):
            result = lang.fetch_stopwords(self.stop_words)
        # Assume collection if not str or None
        else:
            result = frozenset(self.stop_words)
        if self.process_stop_words:
            preprocessor = self.build_preprocessor()
            result = [preprocessor(w) for w in result]
            if self.stemmer == "porter":
                result = lang.porter_stem(result)
            result = frozenset(result)
        return result

    def _validate_params(self):
        super()._validate_params()
        # Check `input`
        valid_input = {"filename", "file", "content"}
        if self.input not in valid_input:
            _invalid_value("input", self.input, valid_input)
        # Check `decode_error`
        valid_decode = {"strict", "ignore", "replace"}
        if self.decode_error not in valid_decode:
            _invalid_value("decode_error", self.decode_error, valid_decode)
        # Check `strip_accents`
        valid_accent = {"ascii", "unicode", None}
        if self.strip_accents not in valid_accent:
            if not callable(self.strip_accents):
                _invalid_value("strip_accents", self.strip_accents, valid_accent)
        # Check `strip_punct`
        if isinstance(self.strip_punct, str):
            if not set(self.strip_punct).issubset(string.punctuation):
                _invalid_value(
                    "strip_punct", self.strip_punct, f"subset of '{string.punctuation}'"
                )
        # Check `stemmer`
        valid_stemmer = {"porter", "wordnet", None}
        if self.stemmer not in valid_stemmer:
            if not callable(self.stemmer):
                _invalid_value("stemmer", self.stemmer, valid_stemmer)
        if self.stemmer == "porter" and self.mark_negation:
            warnings.warn("Porter stemmer may disrupt negation marking.")
        tokenizer_is_default = (
            self.tokenizer is None and self.token_pattern == r"\b\w\w+\b"
        )
        if (
            tokenizer_is_default or self.strip_punct or self.strip_non_word
        ) and self.mark_negation:
            warnings.warn("Sentence punctuation required to mark negation.")


class FreqVectorizer(TfidfVectorizer, VectorizerMixin):
    """Convert a collection of raw documents to a matrix of word-frequency features.

    Extends Scikit-Learn's `TfidfVectorizer` with advanced preprocessing options.
    These include numerous filters, stemming/lemmatization, and markers such as PoS tags.
    Some preprocessing options are applied before tokenization, and some, which require
    tokens, are applied during the tokenization step.

    There are now a wider selection of built-in stopwords sets, and these include the NLTK
    sets for many different languages. Complex stopwords queries are now also supported.


    Parameters
    ----------
    input : {'filename', 'file', 'content'}, default='content'
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be a sequence of items that
        can be of type string or byte.

    encoding : str, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode'}
        Remove accents and perform other character normalization
        during the preprocessing step.
        * 'ascii' is a fast method that only works on characters that have
            an direct ASCII mapping.
        * 'unicode' is a slightly slower method that works on any characters.
        *  None (default) does nothing.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    decode_html_entities : bool
        Decode HTML entities such as '&mdash;' or '&lt;' or '&gt;' into symbols,
        e.g. '—', '<', '>'. True by default.

    lowercase : bool
        Convert all characters to lowercase before tokenizing. True by default.

    strip_extra_space: bool
        Strip extra whitespaces (including tabs and newlines). False by default.

    strip_numeric: bool
        Strip numerals [0-9] from text. False by default.

    strip_non_word: bool
        Strip all non-alphanumeric characters (except underscore). False by default.

    strip_punct: bool or str of punctuation symbols
        If True, strip all punctuation. If passed a string of punctuation symbols, strip
        only those symbols. False by default.

    strip_twitter_handles: bool
        Strip Twitter @mentions. False by default.

    strip_html_tags: bool
        Strip HTML tags such as '<p>' or '<div>'. False by default.

    limit_repeats: bool
        Limit strings of repeating characters, e.g. 'zzzzzzzzzzz', to length 3.

    uniq_char_thresh: float
        Remove tokens with a unique character ratio below threshold. Useful for removing
        repetitive strings like 'AAAAAAAAAAARGH' or 'dogdogdog'. None by default.

    mark_negation: bool
        Mark tokens with '_NEG' which appear between a negation word and sentence
        punctuation. Useful for sentiment analysis. False by default.

    stemmer: {'porter', 'wordnet'}
        Stemming or lemmatization algorithm to use. Both implement caching in order to
        reuse previous computations. Valid options:
        * 'porter' - Porter stemming algorithm (faster).
        * 'wordnet' - Lemmatization using Wordnet (slower).
        * None - Do not stem tokens (default).

    preprocessor : callable, default=None
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
        Only applies if ``analyzer is not callable``.

    tokenizer : callable, default=None
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    analyzer : callable, default=None
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    stop_words : str or list of str
        If a string, it is passed to `tools.language.fetch_stopwords` and
        the appropriate stopword list is returned. Valid strings:
        * 'sklearn_english' - Scikit-Learn's English stopwords.
        * 'nltk_LANGUAGE' - Any NLTK stopwords set, where the fileid (language) follows the underscore.
            For example: 'nltk_english', 'nltk_french', 'nltk_spanish'.
        * Supports complex queries using set operators, e.g. '(nltk_french & nltk_spanish) | sklearn_english'.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    token_pattern : str, default=r"\\b\\w\\w+\\b"
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

        If there is a capturing group in token_pattern then the
        captured group content, not the entire match, becomes the token.
        At most one capturing group is permitted.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
        unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
        only bigrams. Defaults to (1, 1).
        Only applies if ``analyzer is not callable``.

    max_df : float or int
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words). Defaults to 1.0.
        If float in range [0.0, 1.0], the parameter represents a proportion of
        documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float or int
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature. Defaults to 1.
        If float in range of [0.0, 1.0], the parameter represents a proportion
        of documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        None by default.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. None by default.

    binary : bool
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs).
        False by default.

    dtype : dtype
        Type of the matrix returned by fit_transform() or transform().
        'float64' by default.

    norm : {'l2', 'l1'}
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied. None by default.
        * 'l1': Sum of absolute values of vector elements is 1.
        See :func:`preprocessing.normalize`.

    use_idf : bool
        Enable inverse-document-frequency reweighting. False by default.

    smooth_idf : bool
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions. True by default.

    sublinear_tf : bool
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
        False by default.

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.

    fixed_vocabulary_: bool
        True if a fixed vocabulary of term to indices mapping
        is provided by the user

    idf_ : array of shape (n_features,)
        The inverse document frequency (IDF) vector; only defined
        if ``use_idf`` is True.

    stop_words_ : set
        Terms that were ignored because they either:

          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).

        This is only available if no vocabulary was given.
    """

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        decode_html_entities=True,
        lowercase=True,
        strip_extra_space=False,
        strip_numeric=False,
        strip_non_word=False,
        strip_punct=False,
        strip_twitter_handles=False,
        strip_html_tags=False,
        limit_repeats=False,
        uniq_char_thresh=None,
        mark_negation=False,
        stemmer=None,
        preprocessor=None,
        tokenizer=None,
        token_pattern=r"\b\w\w+\b",
        analyzer="word",
        stop_words=None,
        process_stop_words=True,
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm=None,
        use_idf=False,
        smooth_idf=True,
        sublinear_tf=False,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )

        self.decode_html_entities = decode_html_entities
        self.strip_extra_space = strip_extra_space
        self.strip_numeric = strip_numeric
        self.strip_non_word = strip_non_word
        self.strip_punct = strip_punct
        self.strip_twitter_handles = strip_twitter_handles
        self.strip_html_tags = strip_html_tags
        self.limit_repeats = limit_repeats
        self.stemmer = stemmer
        self.uniq_char_thresh = uniq_char_thresh
        self.mark_negation = mark_negation
        self.process_stop_words = process_stop_words

    def get_keywords(self, document, top_n=None):
        check_is_fitted(self, "vocabulary_")
        vec = self.transform([document])
        vocab = utils.swap_index(Series(self.vocabulary_))
        keywords = Series(vec.data, index=vocab.loc[vec.indices], name="keywords")
        if top_n is None:
            top_n = len(keywords)
        return keywords.nlargest(top_n)

    @classmethod
    def from_sklearn(cls, vectorizer, transfer_fit=True):
        allowed_types = (CountVectorizer, TfidfVectorizer, TfidfTransformer)
        if not isinstance(vectorizer, allowed_types):
            raise TypeError(
                f"Expected {[x.__name__ for x in allowed_types]}, got {type(vectorizer).__name__}."
            )
        freq_vec = cls(**vectorizer.get_params())
        if transfer_fit:
            if hasattr(vectorizer, "vocabulary_"):
                freq_vec.vocabulary_ = copy.copy(vectorizer.vocabulary_)
            if hasattr(vectorizer, "fixed_vocabulary_"):
                freq_vec.fixed_vocabulary_ = vectorizer.fixed_vocabulary_
            if hasattr(vectorizer, "stop_words_"):
                freq_vec.stop_words_ = copy.copy(vectorizer.stop_words_)
            if hasattr(vectorizer, "idf_"):
                freq_vec.idf_ = vectorizer.idf_.copy()
        return freq_vec
