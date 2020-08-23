# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 20/08/2020
# Python 3.7.3
# Windows 8
"""Split and preprocess the Gutenberg corpus content."""

import os
import sys

from tqdm import tqdm

# in order to acces module from sister directory
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from lib.author_model import AuthorModel


AUTHORS = ["Anthony Trollope", "Charles Dickens", "Charlotte Mary Yonge",
           "George Alfred Henty", "Henry Rider Haggard",
           "James Fenimore Cooper", "R M Ballantyne", "Robert Louis Stevenson",
           "Sir Walter Scott", "William Dean Howells"]


# function to minimize: difference between sum of subset and goal value
# as this problem is NP-complete we reduce it by looking only at sums of
# slices (=consecutive values)
def opt_slice(array, goal):
    best_slice = (float("inf"), None, None)
    for start in range(len(array)):
        for end in range(start, len(array)):
            diff = abs(sum(array[start:end+1]) - goal)
            if diff < best_slice[0]:
                best_slice = (diff, start, end+1)
    return best_slice[1:]


# filter relevant authors, split them into the three parts
# validation, test and training and preprocess them
def preprocess_gutenberg(source):
    # create directories
    os.mkdir("corpus")
    os.mkdir(os.path.join("corpus", "test"))
    os.mkdir(os.path.join("corpus", "training"))
    # os.mkdir(os.path.join("corpus", "validation"))

    # create mapping of each author to its books
    data = {key: [] for key in AUTHORS}
    for fl in os.listdir(os.path.join(source, "txt")):
        author, title = fl.split("___")
        if author in data:
            data[author].append(fl)
    for author in tqdm(data):
        os.mkdir(os.path.join("corpus", "test", author))
        os.mkdir(os.path.join("corpus", "training", author))
        # os.mkdir(os.path.join("corpus", "validation", author))
        books = sorted(data[author])
        sizes = [os.path.getsize(os.path.join(source, "txt", book)) for book in books]
        val = opt_slice(sizes, sum(sizes)*0.1)
        # books included in the validation set can not be included in the test set
        # their size is set to zero for them not to be used to get any closer to goal value
        test = opt_slice(sizes[:val[0]] + [0]*(val[1] - val[0]) + sizes[val[1]:], sum(sizes)*0.2)
        for i, book in enumerate(tqdm(books, leave=False)):
            if i in range(*val):
                pass
                # AuthorModel.preprocess(os.path.join(source, "txt", book),
                                       # os.path.join("corpus", "validation", author, book))
            elif i in range(*test):
                AuthorModel.preprocess(os.path.join(source, "txt", books[i]),
                                       os.path.join("corpus", "test", author, book))
            else:
                AuthorModel.preprocess(os.path.join(source, "txt", book),
                                       os.path.join("corpus", "training", author, book))


if __name__ == "__main__":
    if os.path.isdir("corpuss"):
        print("The preprocessed splitted corpus is already available.")
    else:
        if len(sys.argv) < 2:
            print("Please pass the path to your unzipped Gutenberg folder.\n")
            print("Synopsis:")
            print("$ python scripts/split_data.py PATH_TO_UNZIPPED_GUTENBERG")
        else:
            if os.path.isdir(sys.argv[1]):
                preprocess_gutenberg(sys.argv[1])
            else:
                raise FileNotFoundError(f"'{sys.argv[1]}' matches no path to a directory.")
