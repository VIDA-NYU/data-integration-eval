import io
import itertools
from datetime import datetime
from pathlib import Path

import math
import numpy as np
import pytz
import torchvision.transforms.functional as TF
from PIL import Image
from matplotlib import pyplot as plt


def get_local_time(time_format='%Y-%m-%d_%H.%M.%S'):
    time_zone = pytz.timezone('Asia/Shanghai')
    ctime = datetime.now(time_zone).strftime(time_format)
    return ctime


def get_project_root():
    return str(Path(__file__).parent.parent.parent)


def one_to_one_matches(matches: dict):
    """
    https://github.com/delftdata/valentine/tree/v1.1
    A filter that takes a dict of column matches and returns a dict of 1 to 1 matches. The filter works in the following
    way: At first it gets the median similarity of the set of the values and removes all matches
    that have a similarity lower than that. Then from what remained it matches columns for me highest similarity
    to the lowest till the columns have at most one match.

    Parameters
    ----------
    matches : dict
        The ranked list of matches

    Returns
    -------
    dict
        The ranked list of matches after the 1 to 1 filter
    """
    matches_value_ls = list(matches.values())

    matches_value_ls_no_dup = [matches_value_ls[0]]
    for i in range(1, len(matches_value_ls)):
        if matches_value_ls[i] != matches_value_ls[i - 1]:
            matches_value_ls_no_dup.append(matches_value_ls[i])

    if len(matches_value_ls_no_dup) < 2:
        return matches

    matched = dict()

    for key in matches.keys():
        matched[key[0]] = False
        matched[key[1]] = False

    median = matches_value_ls_no_dup[math.floor(len(matches_value_ls_no_dup) / 2)]

    matches1to1 = dict()

    for key in matches.keys():
        if (not matched[key[0]]) and (not matched[key[1]]):
            similarity = matches.get(key)
            if similarity >= median:
                matches1to1[key] = similarity
                matched[key[0]] = True
                matched[key[1]] = True
            else:
                break
    return matches1to1


def plot_matrix(sim_matrix, fig_title, xticks, yticks, xlabel, ylabel):
    assert len(xticks) == sim_matrix.shape[1]
    assert len(yticks) == sim_matrix.shape[0]
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(sim_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(fig_title)
    plt.colorbar()

    # tick_marks = np.arange(len(class_names))
    plt.yticks(np.arange(len(yticks)), yticks, rotation=45)
    plt.xticks(np.arange(len(xticks)), xticks, rotation=45, horizontalalignment='right')

    # more classes means small room, so no text
    if len(xticks) <= 12 and len(yticks) <= 12:
        # Use white text if squares are dark; otherwise black.
        threshold = sim_matrix.max() / 2.
        for i, j in itertools.product(range(sim_matrix.shape[0]), range(sim_matrix.shape[1])):
            color = "white" if sim_matrix[i, j] > threshold else "black"
            plt.text(j, i, "{:.2f}".format(sim_matrix[i, j].item()), horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return figure


def plot_to_image(figure):
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    img = Image.open(buf)
    image = TF.to_tensor(img)
    return image
