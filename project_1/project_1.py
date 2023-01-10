#!/usr/bin/env python3
# Jason Duong
# Project #1: Data Dependencies
# October 3, 2022


import re
from itertools import combinations


def extract_vars(instr):
    """
    Parses each simple instruction into their input and output sets.
    Returns a list of of input/output sets ordered by instruction execution.
    """
    parse = lambda f: set(
        re.findall(r"(\w)+", f)
    )  # returns a set of all distinct characters from a string
    return [list(map(parse, i.split("="))) for i in instr]


def check_dependencies(lhs, rhs):
    """
    Returns true if the two input instructions have no data dependencies.
    """
    return all(lhs[i] & rhs[j] == set() for i, j in [[0, 0], [0, 1], [1, 0]])


def calculate(target, chunk):
    """
    Returns a list of any instructions who have no data dependencies with the target instruction
    (i.e. those instrutions that can be parallelizable with the target instruction).
    (i, p) represent indices and variables respectively.
    """
    result = [block[i] for i, p in enumerate(chunk) if check_dependencies(*target, p)]
    print(result if len(result) else "NONE")


def verify(chunk):
    """
    Returns a list of all distinct pairs of instructions that are parallelizable.
    (i, j) and (p, q) represent indicies and variables respectively.
    """
    pairs = combinations(zip(range(len(block)), chunk), 2)
    result = [
        (block[i], block[j]) for (i, p), (j, q) in pairs if check_dependencies(p, q)
    ]
    print(result if len(result) else "NONE")


if __name__ == "__main__":
    input_instr = ["d = b + ( c - d / e )"]
    block = ["b = b * c", "c = c - a", "a = a + b * c"]
    calculate(extract_vars(input_instr), extract_vars(block))  # example 1

    block = ["b = b * c", "d = c - a", "a = a + b * c"]
    verify(extract_vars(block))  # example 2a

    block = ["a = a * b * c", "c = c - a", "a = a + b * c"]
    verify(extract_vars(block))  # example 2b
