#!/usr/bin/env python3
""" that determines if you should stop gradient descent early:"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ that determines if you should stop gradient descent early: """

    if cost < opt_cost - threshold:
        return False, 0
    else:
        count += 1
        if count < patience:
            return False, count
        else:
            return True, count
