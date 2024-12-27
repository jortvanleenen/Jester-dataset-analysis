"""
This script generates statistics of the class counts for our splits.

Author(s): Sana Asghari, Jort van Leenen
License: GNU General Public License v3.0 (GPLv3)
"""
import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def individual_statistics(file: str) -> Counter | None:
    """
    Read a space-seperated split file, give the coefficient of variation and
    return the counts of each class ID in the file.

    :param file: The path to the file to read
    :return: A Counter object with the counts of each class ID in the file
    """
    if not os.path.isfile(file):
        print(f'Error: {file} is not a valid file or cannot be read.')
        return None

    counts = Counter()
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            try:
                number = int(parts[1])
                counts[number] += 1
            except ValueError:
                print(f"Error: Skipping invalid in '{file}': {line.strip()}")
            except IndexError:
                print(f"Error: Skipping invalid in '{file}': {line.strip()}")

    sorted_counts = sorted(counts.values())[:-1]
    mean = np.mean(sorted_counts)
    std = np.std(sorted_counts)
    print(f"CV: {std / mean:.4f} for '{file}'")

    return counts


marker = itertools.cycle(('o', '^', 's'))


def relative_statistics(splits: list[str]) -> None:
    """
    Plot the class counts and print the cosine similarity between the counts.
    
    :param splits: A list of paths to the split files
    """
    plt.figure(figsize=(10, 6))

    max_class = 0
    all_counts = []
    for split in splits:
        counts = individual_statistics(split)
        if counts is None:
            continue

        all_counts.append(counts)

        keys = sorted(counts.keys())
        values = [counts[key] for key in keys]
        max_class = max(max_class, max(keys))

        plt.plot(keys, values, marker=next(marker),
                 label=os.path.basename(split))

    plt.title('Class Counts per Split')
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    plt.xticks(range(max_class + 1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    for i, j in itertools.combinations(range(len(all_counts)), 2):
        a = np.array(list(all_counts[i].values()))
        b = np.array(list(all_counts[j].values()))
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cos_similarity = dot_product / (norm_a * norm_b)
        print(f"Cosine similarity of '{splits[i]}' and '{splits[j]}': "
              f"{cos_similarity:.4f}")


files = ['jester-train.csv', 'jester-validation.csv', 'jester-test.csv']
relative_statistics(files)
