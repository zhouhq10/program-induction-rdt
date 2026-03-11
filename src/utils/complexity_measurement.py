import numpy as np
import textdistance
from collections import Counter
from scipy.stats import wasserstein_distance


def comp_d_hamming(gt, pred):
    return np.count_nonzero(gt != pred)


def comp_d_wasserstein(gt, pred):
    return wasserstein_distance(pred, gt)


def comp_d_levenshtein(gt, pred):
    return textdistance.levenshtein.distance(pred.tolist(), gt.tolist())


def comp_d_n_gram_overlap(seq1, seq2, n=4):
    """
    Calculate the n-gram overlap between two sequences.

    Args:
        seq1 (list or str): The first sequence.
        seq2 (list or str): The second sequence.
        n (int): The length of the n-grams.

    Returns:
        float: The proportion of overlapping n-grams.
    """
    # Generate n-grams for both sequences
    ngrams1 = {tuple(seq1[i : i + n]) for i in range(len(seq1) - n + 1)}
    ngrams2 = {tuple(seq2[i : i + n]) for i in range(len(seq2) - n + 1)}

    # Compute the intersection and union of n-grams
    overlap = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)

    # Calculate overlap proportion
    overlap_score = len(overlap) / len(union) if union else 0.0
    return overlap_score


# Chunk complexity
def count_consecutive_subsequences(sequence):
    if not sequence:
        return []

    lengths = []
    current_length = 1

    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current_length += 1
        else:
            lengths.append(current_length)
            current_length = 1

    # Append the last counted subsequence length
    lengths.append(current_length)

    return lengths


def comp_chunk_complexity(length_sequence):
    complexity = [np.log(length + 1) for length in length_sequence]
    complexity = np.sum(complexity)
    return complexity


def compute_transition_probabilities(sequence):
    # Count occurrences of each element
    counts = Counter(sequence)
    total_count = len(sequence)

    # Compute probability of each element
    p = {k: v / total_count for k, v in counts.items()}

    # Compute transition counts
    transition_counts = Counter(
        (sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)
    )
    transition_total = sum(transition_counts.values())

    # Compute transition probabilities
    p_transition = {
        (k1, k2): v / transition_total for (k1, k2), v in transition_counts.items()
    }

    return p, p_transition


def compute_entropy(sequence):
    p, p_transition = compute_transition_probabilities(sequence)

    entropy = 0
    for x, y in p_transition:
        prob_x = p[x]
        prob_y_given_x = p_transition[(x, y)]

        term = prob_x * prob_y_given_x * (np.log2(prob_x) + np.log2(prob_y_given_x))
        entropy -= term

    return entropy


def count_subsymmetries(sequence):
    """
    Count the number of symmetric (palindromic) subsequences in a sequence.

    Parameters:
    sequence (list): A list of integers or characters.

    Returns:
    int: The number of symmetric subsequences in the sequence.
    """
    n = len(sequence)

    # Create a DP table to store counts of palindromic subsequences
    dp = [[0] * n for _ in range(n)]

    # Every single character is a palindromic subsequence
    for i in range(n):
        dp[i][i] = 1

    # Check for subsequences of length greater than 1
    for length in range(2, n + 1):  # Length of subsequence
        for i in range(n - length + 1):
            j = i + length - 1  # Ending index of the subsequence

            if sequence[i] == sequence[j]:
                dp[i][j] = dp[i + 1][j] + dp[i][j - 1] + 1
            else:
                dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]

    # Sum up the counts of all symmetric subsequences
    total_symmetric_subsequences = sum(dp[i][j] for i in range(n) for j in range(i, n))

    return total_symmetric_subsequences


def lempel_ziv_complexity(sequence):
    """
    Compute the Lempel-Ziv complexity of a given sequence.

    Parameters:
    sequence (list): A list of integers or characters.

    Returns:
    int: The Lempel-Ziv complexity of the sequence.
    """
    n = len(sequence)
    i, complexity = 0, 1  # Start with a complexity of 1 (the first character)
    k, l = 1, 1  # Initial values for k and l

    while i + k <= n:
        if sequence[i : i + k] == sequence[l : l + k]:  # Check if the substring repeats
            k += 1  # If it does, extend the length of the substring
        else:
            if k > 1:  # If k > 1, the substring was repeated
                i += k - 1  # Move the starting point
                complexity += 1  # Increase the complexity count
            else:
                i += 1  # Otherwise, just move one step forward

            l = i + 1  # Reset l and k
            k = 1

    return complexity
