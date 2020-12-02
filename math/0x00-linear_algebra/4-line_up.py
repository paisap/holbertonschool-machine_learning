#!/usr/bin/env python3


def add_arrays(arr1, arr2):
    """ add 2 arrays """
    if len(arr1) != len(arr2):
        return None
    arr = [i + j for i, j in zip(arr1, arr2)]
    return arr
