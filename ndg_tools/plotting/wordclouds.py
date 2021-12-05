from functools import partial, singledispatch
from typing import Dict, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ndg_tools.plotting.utils import get_desat_cmap, smart_subplots
from ndg_tools.typing import SeedLike
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from PIL.Image import Image
from wordcloud import WordCloud

WORDCLOUD_CMAPS = (
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
)


def _sample_cmaps(
    size: int = None, random_state: SeedLike = None
) -> Union[str, np.ndarray]:
    """Randomly select basic sequential cmap(s).

    Parameters
    ----------
    size : int, optional
        Number of colormaps to sample, by default None (single colormap).
    random_state : SeedLike, optional
        Integer, array-like, BitGenerator, Generator or None. Defaults to None.

    Returns
    -------
    str or array of str
        One or more sequential colormap names.
    """
    rng = np.random.default_rng(random_state)
    replace = False
    if size is not None:
        replace = size > len(WORDCLOUD_CMAPS)
    return rng.choice(
        WORDCLOUD_CMAPS,
        size=size,
        replace=replace,
    )


def _make_wordcloud(
    word_scores: Series,
    size: Tuple[int, int],
    cmap: str,
    desat: float,
    render: bool = True,
    **kwargs,
) -> Union[Image, WordCloud]:
    """Generate wordcloud image.

    Parameters
    ----------
    word_scores : Series
        Word frequencies or scores indexed by word.
    size : Tuple[int, int]
        Size of wordcloud. Will me multiplied by 100 to obtain
        pixel size.
    cmap : str
        Recognized name of matplotlib colormap.
    desat : float
        Saturation of colormap.
    render : bool, optional
        If true, render the wordcloud as an image. If false, return
        the Wordcloud object. By default True

    Returns
    -------
    Image or Wordcloud
        PIL Image if `render=True`, otherwise Wordcloud object.
    """
    if (word_scores == 0).all():
        raise ValueError("Word scores cannot all be zero.")
    cmap = get_desat_cmap(cmap, desat=desat)

    # Calculate image size
    width, height = np.array(size) * 100

    cloud = WordCloud(
        colormap=cmap,
        width=width,
        height=height,
        **kwargs,
    )
    cloud.generate_from_frequencies(word_scores)
    return cloud.to_image() if render else cloud


@singledispatch
def wordcloud(
    word_scores: Union[Series, DataFrame],
    *,
    cmap: Union[str, List[str], Dict[str, str]] = "random",
    size: Tuple[float, float] = (5, 3),
    ncols: int = 3,
    desat=0.7,
    ax: Axes = None,
    random_state: SeedLike = None,
    n_jobs: int = None,
    **kwargs,
) -> Union[Axes, Figure]:
    """Plot wordcloud(s) from word frequencies or scores.

    Parameters
    ----------
    word_scores : Series or DataFrame
        Word frequencies or scores indexed by word. Plots multiple wordclouds
        if passed a DataFrame.
    cmap : str, or list of str or dict {cols -> cmaps}, optional
        Name of Matplotlib colormap to use, or 'random' to randomly
        select from basic sequential colormaps. By default 'random'.
    size : tuple of floats, optional
        Size of (each) wordcloud, by default (5, 3).
    ncols : int, optional
        Number of columns, if passing a DataFrame. By default 3.
    ax : Axes, optional
        Axes to plot on, if passing a Series. By default None.
    random_state: SeedLike
        Integer, array-like, BitGenerator, Generator or None. Only used if
        `cmap='random'`. Defaults to None.
    n_jobs : int, optional
        Number of processes for generating wordclouds (DataFrame input only).
        Defaults to None, which is equivalent to 1 or sequential processing.
    Returns
    -------
    Axes or Figure
        Axes of single wordcloud of Figure of multiple wordclouds.
        Returns Axes if `word_scores` is Series, Figure if a DataFrame.
    """
    # This is the dispatch if `word_scores` is neither Series nor DataFrame.
    raise TypeError(f"Expected Series or DataFrame, got {type(word_scores)}")


@wordcloud.register
def _(
    word_scores: Series,
    *,
    cmap: str = "random",
    size: Tuple[float, float] = (5, 3),
    ncols: int = 3,
    desat=0.7,
    ax: Axes = None,
    random_state: int = None,
    n_jobs: int = None,
    **kwargs,
) -> Axes:
    """Dispatch for Series. Returns single Axes with wordcloud."""
    # Create new Axes if none received
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    if cmap == "random":
        cmap = _sample_cmaps(random_state=random_state)
    # Render wordcloud image from scores
    cloud = _make_wordcloud(
        word_scores,
        size,
        cmap,
        desat,
        **kwargs,
    )

    # Display image on `ax`
    ax.imshow(cloud, interpolation="bilinear", aspect="equal")

    if word_scores.name is not None:
        ax.set(title=word_scores.name)

    # Hide grid lines and ticks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


@wordcloud.register
def _(
    word_scores: DataFrame,
    *,
    cmap: Union[str, List[str], Dict[str, str]] = "random",
    size: Tuple[float, float] = (5, 3),
    ncols: int = 3,
    desat=0.7,
    ax: Axes = None,
    random_state: int = None,
    n_jobs: int = None,
    **kwargs,
) -> Figure:
    """Dispatch for DataFrames. Plots each column on a subplot."""
    if ax is not None:
        raise ValueError("`ax` not supported for DataFrame input")

    # Create subplots
    n_clouds = word_scores.shape[1]
    fig, axs = smart_subplots(nplots=n_clouds, size=size, ncols=ncols)

    # Wrangle `cmap` into a dict
    if isinstance(cmap, str):
        if cmap == "random":
            cmap = _sample_cmaps(n_clouds, random_state=random_state)
            cmap = dict(zip(word_scores.columns, cmap))
        else:
            cmap = dict.fromkeys(word_scores.columns, cmap)
    elif isinstance(cmap, list):
        cmap = dict(zip(word_scores.columns, cmap))
    elif not isinstance(cmap, dict):
        raise TypeError("`cmap` must be str, list, or dict {cols -> cmaps}")

    # Plot each column
    _make_wordcloud_d = joblib.delayed(
        partial(_make_wordcloud, size=size, desat=desat, **kwargs)
    )

    clouds = joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
        _make_wordcloud_d(word_scores.loc[:, column], cmap=cmap[column])
        for column in word_scores.columns
    )

    for cloud, column, ax in zip(clouds, word_scores.columns, axs.flat):
        # Display image on Axes
        ax.imshow(cloud, interpolation="bilinear", aspect="equal")
        ax.set_title(column)

        # Hide grid lines and ticks
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i, ax in enumerate(axs.flat):
        if i >= word_scores.columns.size:
            ax.set_visible(False)

    fig.tight_layout()
    return fig
