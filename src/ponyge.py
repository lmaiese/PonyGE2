#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """

from utilities.algorithm.general import check_python_version

check_python_version()

from stats.stats import get_stats
from algorithm.parameters import params, set_params
import sys


def test_function():
    pass


def mane():
    """ Run program """
    set_params(sys.argv[1:])  # exclude the ponyge.py arg itself

    print(params['POPULATION_SIZE'], params['GENERATIONS'], params['TOURNAMENT_SIZE'])

    # Run evolution
    individuals = params['SEARCH_LOOP']()

    # Print final review
    get_stats(individuals, end=True)

    print(individuals.__class__)

    print(individuals)


if __name__ == "__main__":
    mane()
