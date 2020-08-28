# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 28/08/2020
# Python 3.7.3
# Windows 8
"""MutableMappings for frequency distributions."""

from collections.abc import MutableMapping, Mapping, Iterable
import logging

import matplotlib.pyplot as plt
import numpy as np

from lib.errors import ScarceDataError, log_exception


LOG = logging.getLogger(__name__)  # module logger


class Distribution(MutableMapping):
    """A frequency distribution to record event outcomes.

    A distribution maps events to the number of times they have
    been observed. It is similar to a dictionary but in so far
    different that values may only be of type integer.

    Args:
        iterable(Optional[Iterable]): When given,
            initialize the distribution with its items.

    Attributes:
        _distr(dict): The mapping.
        _total(int): Total number of items in the distribution.
    """
    def __init__(self, iterable=None):
        self._distr = dict()
        self._total = 0

        if iterable is not None:
            self.update(iterable)

    @property
    def distr(self):
        """Return the recorded distribution as a dictionary."""
        return self._distr

    @property
    def total(self):
        """
        Return the total number of observations that have been
        recorded by this distribution.
        """
        return self._total

    @log_exception(LOG)
    def prob_dist(self, iterable=None):
        """Calculate probability distribution over all observations.

        Args:
            iterable: Specify to only calculate the probability
                for a subset of observations, normalization
                is still performed with the total number
                of observations in the whole distribution.

        Returns:
            dict: Dict with normalized counts.
        """
        if self._total < 1:
            raise ScarceDataError("Method 'prob_dist' needs at least one data point.")
        if iterable is None:
            iterable = self.keys()
        return {k: self[k]/self._total for k in iterable}

    @log_exception(LOG)
    def plot(self, title, iterable=None):
        """
        Plot samples from the probability distribution over
        all observations as a pie chart. Summarizing all items
        with probabilities smaller 0.02 to <other>.

        Args:
            title(str): Title to be displayed over the chart.
            iterable(Optional[]): Specify to only include a subset
                of observations. If not specified all are plotted.
        """
        if not iterable:
            iterable = self.keys()
            if not iterable:
                raise ScarceDataError("Method 'plot' needs at least one data point.")
        else:
            # because we need to iterate more than once
            if iter(iterable) is iterable:
                iterable = list(iterable)
        # organize data
        probs = self.prob_dist(iterable)
        slices = sorted(iterable, key=lambda x: probs[x], reverse=True)
        sizes = [probs[slc] for slc in slices]
        # filter scarce data points
        for i in range(len(slices)):
            if sizes[i] < 0.02:
                slices = slices[:i] + ['<other>']
                sizes = sizes[:i] + [sum(sizes[i:])]
                break
        # plot settings
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=["{:.2%}".format(size) for size in sizes],
               shadow=False, startangle=90)
        ax.axis('equal')
        plt.title(title)
        plt.legend(slices, loc=3)
        plt.show()

    @log_exception(LOG)
    def update(self, iterable):
        if isinstance(iterable, Mapping):
            if any(map(lambda x: not isinstance(x, int) or x < 0, iterable.values())):
                raise TypeError("Values of Distribution need to be positive integers.")
            # called after initialization
            if self._distr:
                self_get = self.get
                for obsv, times in iterable.items():
                    self[obsv] = times + self_get(obsv)
            # called in __init__
            else:
                self._total = sum(iterable.values())
                self._distr.update(iterable)  # faster than adding one at a time
        elif isinstance(iterable, Iterable):
            for item in iterable:
                self[item] += 1

#################
# private methods
#################

    def __getitem__(self, key):
        # does not raise a KeyError if event not yet observed
        return self._distr.get(key, 0)

    def __setitem__(self, key, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("Values of Distribution need to be positive integers.")
        self._total += (value - self[key])
        self._distr[key] = value

    def __delitem__(self, key):
        if key in self._distr:
            self._total -= self._distr[key]
            del self._distr[key]

    def __iter__(self):
        return iter(self._distr)

    def __len__(self):
        return len(self._distr)


class IntegerDistribution(Distribution):
    """Subclass of Distribution.

    See DocString for Distribution.
    Additionally is guaranteed to contain
    only keys of type integer.

    Args:
        iterable(Optional[Iterable]): When given,
            initialize the distribution with its items.

    Attributes:
        _distr(dict): The mapping.
        _total(int): Total number of items in the distribution.
    """
    @log_exception(LOG)
    def mean(self):
        """Sample arithmetic mean (average).

        Returns:
            float: Sum of recorded observations
                divided by the number of observations.
        """
        if self._total < 1:
            raise ScarceDataError("Method 'mean' needs at least one data point.")
        return sum(map(lambda x: x[0]*x[1], self.items()))/self._total

    @log_exception(LOG)
    def var(self, m=None):
        """Corrected sample variance.

        Args:
            m(Optional[float]): Precalculated mean if existing.

        Returns:
            float: Measure of the average degree to which
                each observation differs from the mean.
        """
        if self._total < 2:
            raise ScarceDataError("Method 'mean' needs at least two data points.")
        if m is None:
            m = self.mean()
        return sum(map(lambda x: self[x]*(x - m)**2, self))/(self._total - 1)

    @log_exception(LOG)
    def stdev(self, m=None):
        """Corrected sample standard deviation.

        Args:
            m(Optional[float]): Precalculated mean if existing.

        Returns:
            float: Measure how far observations are
                spread out from their average value.
        """
        return self.var(m)**0.5

    @log_exception(LOG)
    def plot(self, title, label, start=None, end=None, steps=None, iterable=None):
        """
        Plot samples from the probability distribution over
        all observations in ascending order as a
        bar chart.

        Args:
            title(str): Title to be displayed over the plot.
            label(str): Description of the bars in the legend.
            start(Optional[int]): x-Axis left border.
            end(Optional[int]): x-Axis right border.
            steps(Optional[int]): Distance to put x-Axis ticks at.
            iterable(Optional[]): Specify to only include a subset
                of observations. If not specified all are plotted.
        """
        if not iterable:
            iterable = self.keys()
            if not iterable:
                raise ScarceDataError("Method 'plot' requires at least one data point.")
        else:
            # because we need to iterate more than once
            if iter(iterable) is iterable:
                iterable = list(iterable)
        # create plot
        fig, ax = plt.subplots()
        # organize data
        bars = sorted(iterable)
        if start is None:
            start = bars[0]
        if end is None:
            end = bars[-1]
        if steps is None:
            steps = max(1, (start + end)//10)
        bars = [bar for bar in bars if start <= bar <= end]
        heights = [self.prob_dist(iterable)[k] for k in bars]
        # set axis settings
        ax.set_axisbelow(True)
        ax.set_title(title)
        ax.set_xticks(np.arange(start, end + 1, steps))
        ax.set_ylabel('relative frequency')
        # remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # add major gridlines in the background
        ax.grid(color='white', linestyle='-', linewidth=2, alpha=0.5)
        ax.set_facecolor((0.95, 0.95, 0.95))
        # draw bars
        ax.bar(bars, heights, color='mediumseagreen', label=label, alpha=0.60)
        ax.plot(bars, heights, color='mediumseagreen')
        ax.legend()
        plt.show()

    @log_exception(LOG)
    def update(self, iterable):
        if any(map(lambda x: not isinstance(x, int), iterable)):
            raise TypeError("Keys of IntegerDistribution need to be integers.")
        super().update(iterable)

#################
# private methods
#################

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError("Keys of IntegerDistribution have to be integers.")
        super().__setitem__(key, value)
