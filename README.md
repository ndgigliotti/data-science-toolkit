# ndg-tools

I developed the tools in this package for my own use on data science projects. They are in an early stage of development and not intended for public use. Nevertheless, you are free to install them.

```
pip install git+https://github.com/ndgigliotti/ndg-tools.git@main
```
# Important Modules

## `language`

One of the most extensive modules in the package is `ndg_tools.language`, which contains string processing functions, token processing functions, n-gram search functions, and other utilities.

### `language.processors`

A number of text (`str`) and token (`list` of `str`) processing functions can be found in `ndg_tools.language.processors`. For example, `strip_html_tags` thoroughly removes HTML tags from strings. All of the text processing functions are **polymorphic**, meaning that they behave differently according to the type of object passed to them. They work on singular strings as well as string-containers such as `list`s, `set`s, `ndarray`s, `Series`, or `DataFrame`s.

All of the text and token processors feature optional **parallel processing**, implemented with [joblib](https://joblib.readthedocs.io/en/latest/). Parallel processing can be activated using the `n_jobs` keyword argument. For example, passing `n_jobs=-1` sets the number of jobs to the number of available cores. In a future update, the batch size will also be adjustable in every function.

All of the text and token processors have a built-in [tqdm](https://github.com/tqdm/tqdm) progress bar which will appear in Jupyter Notebooks. The progress bar only appears when the function is used on a string container object.

Several functions can be chained together in a pipeline using the `make_preprocessor` function. The function chain will be applied to each string independently, no matter what object type is passed.

### `language.ngrams`

Inside `ndg_tools.language.ngrams` are functional wrappers of the [nltk](https://www.nltk.org/) collocation search tools. There is also a `stratified_ngrams` function, which allows for collocation-searching relative to a categorical variable. In other words, the documents are grouped by a categorical variable and each group is scanned independently for collocations.

## `sklearn`

The `ndg_tools.sklearn` module contains tools related to Scikit-Learn and the Scikit-Learn API. Currently it contains a universal hyperparameter optimization function called `sweep`, and some custom text vectorizers such as `VaderVectorizer` and `FreqVectorizer`.

### `sklearn.selection.sweep`

The purpose of the `sweep` function is to provide a flexible interface to all of Scikit-Learn's hyperparameter optimization objects. The function itself has many parameters, which some may see as excessively complex. However, there are a number of benefits:

- Easily switch between different search types by setting `kind` to 'grid', 'rand', 'hgrid', or 'hrand'.
- Immediately pickle the search estimator or CV results when the search is finished.
- Add a prefix such as 'col__vec__' to the entire hyperparameter space specification.
- Specify the hyperparameter space with `Series` objects in addition to `list`s and `dict`s.
- Hold some parameters in place for repeat use with `functools.partial`.

### `sklearn.vectorizers.VaderVectorizer`

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analysis tool which assigns polarity scores to a short piece of text. While NLTK provides a nice implementation of VADER, I like to integrate my feature extraction/engineering with the Scikit-Learn API so it can be tuned and tested alongside the model. `VaderVectorizer` is simply a text vectorizer which extracts the four VADER polarity scores: negative, neutral, positive, and compound. It is intended to be used in a `FeatureUnion` with another vectorizer.

### `sklearn.vectorizers.FreqVectorizer`

Because Scikit-Learn's `CountVectorizer` and `TfidfVectorizer` offer only a few preprocessing options, I extended them with many more. Sometimes preprocessing steps must be taken to avoid overfitting, but sometimes they are open to optimization. The new preprocessing options utilize functions in the `ndg_tools.language` module.

One new preprocessing step which is enabled by default is decoding HTML entities like `&amp;` or `&mdash;`. These codes for punctuation symbols appear frequently in text, and are liable to become noise if not decoded.

Another preprocessing option for use in sentiment analysis is negation-marking. Setting `mark_negation=True`, tells the vectorizer to mark terms '_NEG' which appear between a negation term and sentential punctuation.

Note that while `FreqVectorizer` is a subclass of `TfidfVectorizer`, it is set to act as a `CountVectorizer` by default. IDF weighting, binarization, normalization, and other options can be easily activated.

## `outliers`

The `ndg_tools.outliers` module provides a number of tools for dealing with outliers. The three basic approaches to outlier detection in that module are the IQR proximity rule, the Z-score method (i.e. standard score), and simple quantile-based detection. 

The detection methods are combined with ways of dealing with outliers once detected. There are two different options: trimming or Winsorization. Trimming is the traditional approach of dropping outliers from the dataset. Winsorization, on the other hand, means resetting each outlying value to the value of the nearest inlier. Winsorization reduces the influence of outliers without throwing away entire observations (i.e. rows of the table).

# To Do

 - [ ] Write more tests to attain complete test coverage of the package.
 - [ ] Clean up and standardize `ndg_tools.plotting`.
 - [ ] Develop a subset of the tools into a specialized package for public use.
