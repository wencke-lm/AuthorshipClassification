# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 19/08/2020
# Python 3.7.3
# Windows 8
"""Counter subclasses for distribution."""

from collections import Counter
import logging

import matplotlib.pyplot as plt
import numpy as np

from scripts.errors import ScarceDataError, log_exception


LOG = logging.getLogger(__name__)  # module logger


class Distribution(Counter):
    """Subclass of collections.Counter for storing distributions
        of hashable items.

    Args:
        iterable: When given, initialize with its items.

    Attributes:
        _total(int): Total number of items in the distribution.
    """
    def __init__(self, iterable=''):
        self._total = 0

        for item in iterable:
            self.inc(item)

    def inc(self, item):
        """Update distribution.

        Args:
            item(hashable type): Increase count of 'item' by one,
                after adding it if not yet present.
        """
        self._total += 1
        self[item] += 1

    @log_exception(LOG)
    def prob_dist(self, iterable=None):
        """Calculate the relative frequency of all items.
        
        Args:
            iterable: Specify to only include a subset of items.

        Returns:
            dict: Probability distribution over items.
        """
        if self._total < 1:
            raise ScarceDataError("method 'prob_dist' needs atleast one data point")
        if iterable is None:
            iterable = self.keys()
        return {k:self[k]/self._total for k in iterable}

    def total(self):
        """Getter for attribute ._total."""
        return self._total

    @log_exception(LOG)
    def plot(self, title, iterable=None):
        """Plot samples from the probability distribution as a
            pie chart. Summarizing all items with probabilities
            smaller 0.02 to <other>.

        Args:
            title(str): Title to be displayed over the chart.
            iterable: Specify to only include a subset of items.
                If not specified all items are plotted.
        """
        if not iterable:
            iterable = self.keys()
            if not iterable:
                raise ScarceDataError("method 'plot' needs atleast one data point")
        probs = self.prob_dist(iterable)
        slices = sorted(iterable, key=lambda x:probs[x], reverse=True)
        sizes = [probs[slc] for slc in slices]
        for i, x in enumerate(slices):
            if sizes[i] < 0.02:
                slices = slices[:i] + ['<other>']
                sizes = sizes[:i] + [sum(sizes[i:])]
                break
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=["{:.2%}".format(v) for v in sizes], shadow=False, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(title)
        plt.legend(slices, loc=3)
        plt.show()


class IntegerDistribution(Distribution):
    @log_exception(LOG)
    def inc(self,item):
        """Update distribution.

        Args:
            item(int): Increase count of 'item' by one,
                after adding it if not yet present.
        """
        if not isinstance(item, int):
            raise TypeError("items have to be of type integer")
        super().inc(item)

    @log_exception(LOG)
    def mean(self):
        """
        Returns:
            float: Sample arithmetic mean.
        """
        if self._total < 1:
            raise ScarceDataError("method 'mean' needs atleast one data point")
        return sum(map(lambda x: x[0]*x[1], self.items()))/self._total

    @log_exception(LOG)
    def var(self, m=None):
        """
        Args:
            m(int): Precalculated mean if existing.

        Returns:
            float: Corrected sample variance.
        """
        if self._total < 2:
            raise ScarceDataError("method 'mean' needs atleast two data points")
        if m is None:
            m = self.mean()
        return sum(map(lambda x: (x - m)**2, self.elements()))/(self._total - 1)

    def stdev(self, m=None):
        """
        Args:
            m(int): Precalculated mean if existing.

        Returns:
            float: Corrected sample standard devaíation.
        """
        return self.var(m)**0.5

    def plot(self, title, label, start=None, end=None, steps=None,iterable=None):
        """Plot samples from the probability distribution as a
            bar chart.
        
        Args:
            title(str): Title to be displayed over the plot.
            label(str): Description of the bars in the legend.
            start(int): x-Axis left border.
            end(int): x-Axis right border.
            steps(int): Distance to put x-Axis ticks at.
            iterable: Specify to only include a subset of items.
                If not specified all items are plotted.
        """
        if not iterable:
            iterable = self.keys()
            if not iterable:
                raise ScarceDataError("method 'plot' needs atleast one data point")
        # organize data
        bars = sorted(iterable)
        heights = [self.prob_dist(iterable)[k] for k in bars]
        # create plot
        fig, ax = plt.subplots()
        # set axis settings
        if start is None:
            start = bars[0]
        if end is None:
            end = bars[-1]
        if steps is None:
            steps = max(1,(start+end)//10)
        ax.set_axisbelow(True)
        ax.set_title(title)
        ax.set_xticks(np.arange(start, end+1, steps))
        ax.set_ylabel('relative frequency')
        # remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # add major gridlines in the background
        ax.grid(color='white', linestyle='-', linewidth=2, alpha=0.5)
        ax.set_facecolor((0.95,0.95,0.95))
        # draw bars
        ax.bar(bars, heights, color='mediumseagreen', label=label, alpha=0.60)
        ax.plot(bars, heights, color='mediumseagreen')
        ax.legend()
        plt.show()
