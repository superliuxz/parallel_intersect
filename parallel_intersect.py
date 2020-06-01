import random  # Do NOT seed the random generator
import unittest
from typing import Tuple, Iterator


def get_value(array, idx):
  # If index is out of right boundary, assume +INF
  # If index is out of left boundary, assume -INF
  if idx < 0:
    return -float('inf')
  elif idx >= len(array):
    return float("inf")
  return array[idx]


def create_disjoint_sublist(
    A: list, B: list, i: int, p: int) -> Tuple[int, int]:
  if len(B) > len(A):
    i, j = create_disjoint_sublist(B, A, i, p)
    return j, i

  length = (len(A) + len(B)) // p
  diag = i * length

  if diag > len(A):
    a_top = len(A)
    b_top = diag - len(A)
  else:
    a_top = diag
    b_top = 0
  a_btm = b_top
  while True:
    ai = (a_top + a_btm) // 2
    bi = diag - ai
    a_val = get_value(A, ai)
    b_val = get_value(B, bi - 1)
    if a_val > b_val:
      a_val = get_value(A, ai - 1)
      b_val = get_value(B, bi)
      if a_val < b_val:
        return ai, bi
      # Since the adjacency list has not duplicates, this branch can only occur
      # at most once per function call.
      elif a_val == b_val:
        diag += 1
      else:
        a_top = ai - 1
    else:
      a_btm = ai + 1


def partition(A: list, B: list, p: int) -> Iterator[list]:
  """Partition array A and B into p disjointed sets.

  A = {A0, A1, ... Ap-1}, B = {B0, B1, ... Bp-1},
  where max(A0, B0) < min (A1, B1), max(A1, B1) < min(A2, B2) ...
        max(Ap-2, Bp-2) < min(Ap-1, Bp-1)

  Such partitioning guarantees that the intersection on A and B can be
  parallelized to intersect(A1, B1) union intersect(A2, B2) ...
  union intersect(Ap-1, Bp-1).
  """
  a_start, b_start = 0, 0
  for i in range(1, p):
    a_start_now, b_start_now = create_disjoint_sublist(A, B, i, p)
    yield A[a_start: a_start_now], B[b_start: b_start_now]
    a_start, b_start = a_start_now, b_start_now
  yield A[a_start:], B[b_start:]


_TEST_CASE = {
  # all unique elements
  "unique1": {"p": 2,
              "arr1": [1, 2, 3, 5, 8, 9],
              "arr2": [4, 6, 7, 11, 13],
              "expected": [([1, 2, 3, 5], [4]),
                           ([8, 9], [6, 7, 11, 13])]},
  "unique2": {"p": 3,
              "arr1": [1, 2, 3, 5, 8, 9],
              "arr2": [4, 6, 7, 11, 13],
              "expected": [([1, 2, 3], []),
                           ([5], [4, 6]),
                           ([8, 9], [7, 11, 13])]},
  "unique3": {"p": 4,
              "arr1": [1, 2, 3, 5, 8, 9],
              "arr2": [4, 6, 7, 11, 13],
              "expected": [([1, 2], []),
                           ([3], [4]),
                           ([5], [6]),
                           ([8, 9], [7, 11, 13])]},
  "unique4": {"p": 5,
              "arr1": [1, 2, 3, 5, 8, 9],
              "arr2": [4, 6, 7, 11, 13],
              "expected": [([1, 2], []),
                           ([3], [4]),
                           ([5], [6]),
                           ([8], [7]),
                           ([9], [11, 13])]},
  # one partition
  "one_partition": {"p": 1,
                    "arr1": [1, 2, 3, 5, 8, 9],
                    "arr2": [4, 6, 7, 11, 13],
                    "expected": [([1, 2, 3, 5, 8, 9], [4, 6, 7, 11, 13])]},
  # A is longer than B
  "long_short1": {"p": 2,
                  "arr1": [1, 2, 3, 5, 8, 9],
                  "arr2": [4],
                  "expected": [([1, 2, 3], []),
                               ([5, 8, 9], [4])]},
  "long_short2": {"p": 4,
                  "arr1": [1, 2, 3, 5, 8, 9],
                  "arr2": [4],
                  "expected": [([1], []),
                               ([2], []),
                               ([3], []),
                               ([5, 8, 9], [4])]},
  # B is longer than A
  "short_long1": {"p": 2,
                  "arr1": [8],
                  "arr2": [4, 6, 7, 11, 13],
                  "expected": [([], [4, 6, 7]),
                               ([8], [11, 13])]},
  "short_long2": {"p": 3,
                  "arr1": [8],
                  "arr2": [4, 6, 7, 11, 13],
                  "expected": [([], [4, 6]),
                               ([8], [7]),
                               ([], [11, 13])]},
  # The equal element must be placed within the same partition.
  "equal_ele1": {"p": 2,
                 "arr1": [1, 2, 3, 7, 8, 9],
                 "arr2": [4, 6, 7, 11, 13],
                 "expected": [([1, 2, 3], [4, 6]),
                              ([7, 8, 9], [7, 11, 13])]},
  "equal_ele2": {"p": 3,
                 "arr1": [1, 2, 3, 7, 8, 9],
                 "arr2": [4, 6, 7, 11, 13],
                 "expected": [([1, 2, 3], []),
                              ([7], [4, 6, 7]),
                              ([8, 9], [11, 13])]},
  "equal_ele3": {"p": 4,
                 "arr1": [1, 2, 3, 7, 8, 9],
                 "arr2": [4, 6, 7, 11, 13],
                 "expected": [([1, 2], []),
                              ([3], [4]),
                              ([7], [6, 7]),
                              ([8, 9], [11, 13])]},
  # The equal element is at the beginning of list B
  "equal_ele_4": {"p": 2,
                  "arr1": [1, 2, 3, 7, 8, 9],
                  "arr2": [7, 10, 11, 13],
                  "expected": [([1, 2, 3, 7], [7]),
                               ([8, 9], [10, 11, 13])]},
  "equal_ele_5": {"p": 4,
                  "arr1": [1, 2, 3, 7, 8, 9],
                  "arr2": [7, 10, 11, 13],
                  "expected": [([1, 2], []),
                               ([3, 7], [7]),
                               ([8], []),
                               ([9], [10, 11, 13])]},
  # The equal element is at the end of list A
  "equal_ele_6": {"p": 2,
                  "arr1": [1, 2, 3, 7, 8, 9],
                  "arr2": [4, 5, 6, 9, 10, 11, 13],
                  "expected": [([1, 2, 3], [4, 5, 6]),
                               ([7, 8, 9], [9, 10, 11, 13])]},
  "equal_ele_7": {"p": 4,
                  "arr1": [1, 2, 3, 7, 8, 9],
                  "arr2": [4, 5, 6, 9, 10, 11, 13],
                  "expected": [([1, 2, 3], []),
                               ([], [4, 5, 6]),
                               ([7, 8, 9], [9]),
                               ([], [10, 11, 13])]},
  # Multiple equal elements
  "equal_ele_8": {"p": 2,
                  "arr1": [1, 2, 3, 7, 8, 9, 11],
                  "arr2": [2, 4, 6, 9, 10, 11, 13],
                  "expected": [([1, 2, 3, 7], [2, 4, 6]),
                               ([8, 9, 11], [9, 10, 11, 13])]},
  "equal_ele_9": {"p": 3,
                  "arr1": [1, 2, 3, 7, 8, 9, 11],
                  "arr2": [2, 4, 6, 9, 10, 11, 13],
                  "expected": [([1, 2, 3], [2]),
                               ([7, 8], [4, 6]),
                               ([9, 11], [9, 10, 11, 13])]},
  "equal_ele_10": {"p": 4,
                   "arr1": [1, 2, 3, 7, 8, 9, 11],
                   "arr2": [2, 4, 6, 9, 10, 11, 13],
                   "expected": [([1, 2], [2]),
                                ([3], [4, 6]),
                                ([7, 8, 9], [9]),
                                ([11], [10, 11, 13])]},
  "equal_ele_11": {"p": 5,
                   "arr1": [1, 2, 3, 7, 8, 9, 11],
                   "arr2": [2, 4, 6, 9, 10, 11, 13],
                   "expected": [([1, 2], [2]),
                                ([3], []),
                                ([], [4, 6]),
                                ([7, 8], []),
                                ([9, 11], [9, 10, 11, 13])]},
}


class TestParallelIntersect(unittest.TestCase):

  def test_handwritten_cases(self):
    for case_label, data in _TEST_CASE.items():
      self.assertEqual(data["expected"],
                       list(partition(data["arr1"], data["arr2"], data["p"])))

  def test_random_cases(self):
    for i in range(1000):
      # Create two sorted lists. Since the algorithm is designed for adjacency
      # list, we use |random.sample| to ensure the list itself does not contain
      # duplicated values.
      la = sorted(random.sample(range(0, 2000), 1000))
      lb = sorted(random.sample(range(0, 2000), 600))
      # Intersection using Python set
      expected = sorted(set(la).intersection(set(lb)))
      actual = list()
      # Iterate through the disjoint sub-lists and intersect the two sub-lists
      for sub_la, sub_lb in partition(la, lb, 10):
        actual.extend(sorted(set(sub_la).intersection(set(sub_lb))))
      self.assertEqual(expected, sorted(list(actual)))


if __name__ == '__main__':
  unittest.main()
