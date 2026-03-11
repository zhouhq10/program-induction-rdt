import string, pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def extract_obj(file):
    file = open(file, "rb")
    obj_file = pickle.load(file)
    file.close()
    return obj_file


def save_obj(base_path, step_name, index, data, filename_suffix):
    # Create the full file path
    filepath = f"{base_path}/{step_name}/{index}_{filename_suffix}.obj"

    # Open the file and save the data
    with open(filepath, "wb") as filehandler:
        pickle.dump(data, filehandler)


def msg2dict(msg):
    result = dict()
    if "note_on" in msg:
        on_ = True
    elif "note_off" in msg:
        on_ = False
    else:
        on_ = None
    result["time"] = int(
        msg[msg.rfind("time") :]
        .split(" ")[0]
        .split("=")[1]
        .translate(str.maketrans({a: None for a in string.punctuation}))
    )

    if on_ is not None:
        for k in ["note", "velocity"]:
            result[k] = int(
                msg[msg.rfind(k) :]
                .split(" ")[0]
                .split("=")[1]
                .translate(str.maketrans({a: None for a in string.punctuation}))
            )
    return [result, on_]


# Function to calculate the transition matrix for any order
def calculate_transition_matrix(sequences, num_notes=6, order=1):
    # Initialize the transition matrix with zeros
    transition_matrix = np.zeros((num_notes, num_notes))

    # Iterate over each melody sequence (array)
    for sequence in sequences:
        # Loop through each pair of notes with the specified gap (order)
        for i in range(len(sequence) - order):
            current_note = sequence[i] - 1  # Current note (adjust for 0-indexing)
            next_note = sequence[i + order] - 1  # Next note with gap (order)
            transition_matrix[current_note, next_note] += 1

    # Normalize the matrix to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        transition_matrix,
        row_sums,
        out=np.zeros_like(transition_matrix),
        where=row_sums != 0,
    )

    return transition_matrix


def reassign_values(arr):
    # Sort the array and create a dictionary to map each value to its rank
    sorted_arr = list(set(sorted(arr)))
    value_to_rank = {val: rank + 1 for rank, val in enumerate(sorted_arr)}

    # Reassign values in the original array based on the rank
    reassigned_arr = [value_to_rank[val] for val in arr]

    return reassigned_arr


# # Example usage
# arr = [9, 8, 85, 4, 3]
# reassigned_arr = reassign_values(arr)
# print(reassigned_arr)  # Output: [4, 3, 5, 2, 1]

import numpy as np
from scipy import stats


def calculate_length_statistics(all_pressed_notes):
    # Calculate melody length for each array
    melody_length = [len(array) for array in all_pressed_notes]

    # Mean
    mean = np.mean(melody_length)

    # Median
    median = np.median(melody_length)

    # Mode
    mode = stats.mode(melody_length)[0][0]

    # Standard Deviation
    std_dev = np.std(melody_length)

    # Variance
    variance = np.var(melody_length)

    # Minimum and Maximum
    min_value = np.min(melody_length)
    max_value = np.max(melody_length)

    # Sum
    sum_value = np.sum(melody_length)

    # Percentiles (25th, 50th, and 75th percentiles)
    percentiles = np.percentile(melody_length, [25, 50, 75])

    # Print results
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Variance: {variance}")
    print(f"Min: {min_value}")
    print(f"Max: {max_value}")
    print(f"Sum: {sum_value}")
    print(f"25th Percentile: {percentiles[0]}")
    print(f"50th Percentile: {percentiles[1]}")
    print(f"75th Percentile: {percentiles[2]}")


# Example usage:
# all_pressed_notes = [...]  # Replace with your list of note sequences
# calculate_statistics(all_pressed_notes)


# Function to plot the transition matrix
def plot_transition_matrix(transition_matrix, num_notes=6, order=1):
    plt.figure(figsize=(8, 6))
    plt.imshow(transition_matrix, cmap="Blues", interpolation="none")
    plt.colorbar(label="Probability")
    plt.title(f"Transition Distribution Matrix (Order {order})")
    plt.xlabel("To Note")
    plt.ylabel("From Note")

    # Adding labels to the matrix
    plt.xticks(ticks=np.arange(num_notes), labels=np.arange(1, num_notes + 1))
    plt.yticks(ticks=np.arange(num_notes), labels=np.arange(1, num_notes + 1))

    # Annotate each cell with the corresponding probability
    for i in range(num_notes):
        for j in range(num_notes):
            plt.text(
                j,
                i,
                f"{transition_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    plt.show()


def plot_midi_instruments(instruments):
    """
    Analyze and plot the distribution of MIDI instruments.

    Parameters:
    instruments (list): A list of MIDI instrument numbers.

    Returns:
    tuple: (figure, unique_instrument_count)
    """
    # Count the number of unique instruments
    unique_instruments = len(set(instruments))
    print(f"Number of different instruments: {unique_instruments}")

    # Count the occurrences of each instrument
    instrument_counts = Counter(instruments)

    # Create a histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        instruments, bins=range(0, 129, 8), align="left", rwidth=0.8, edgecolor="black"
    )

    # Customize the plot
    ax.set_title("Distribution of MIDI Instruments", fontsize=16)
    ax.set_xlabel("Instrument Number", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xticks(range(0, 129, 16))
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Add grid for better readability
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate the most common instrument
    most_common = instrument_counts.most_common(1)[0]
    ax.annotate(
        f"Most common: {most_common[0]} (x{most_common[1]})",
        xy=(most_common[0], most_common[1]),
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    plt.tight_layout()

    return fig, unique_instruments


# Example usage:
# instruments = [your list of MIDI instrument numbers]
# fig, unique_count = analyze_midi_instruments(instruments)
# plt.show()


def plot_melody_note_seq(melody, figsize=(10, 3), index=None):
    """
    Create a scatter plot of a melody.

    Parameters:
    melody (array-like): The melody to plot.
    figsize (tuple): Figure size (width, height) in inches. Default is (10, 3).
    index (int, optional): Index of the melody (for title). If None, no index is shown.

    Returns:
    fig, ax: The created figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize)

    melody_array = np.array(melody)

    ax.scatter(np.arange(len(melody_array)), melody_array, marker="s", s=1)

    ax.set_xlabel("Time")
    ax.set_ylabel("Note")

    if index is not None:
        ax.set_title(f"Melody Scatter Plot (Index: {index})")
    else:
        ax.set_title("Melody Scatter Plot")

    plt.tight_layout()
    return fig, ax


def plot_distinct_notes_histogram(melodies, figsize=(10, 6), x_tick_interval=10):
    """
    Create a histogram of distinct notes for a set of melodies.

    Parameters:
    melodies (list of array-like): List of melodies to analyze.
    figsize (tuple): Figure size (width, height) in inches. Default is (10, 6).
    x_tick_interval (int): Interval for x-axis ticks. Default is 10.

    Returns:
    fig, ax: The created figure and axes objects.
    """
    # Count distinct notes in each melody
    distinct_counts = [len(np.unique(melody)) for melody in melodies]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the histogram
    ax.hist(
        distinct_counts,
        bins=range(1, max(distinct_counts) + 2),
        edgecolor="black",
        align="left",
    )

    # Set labels and title
    ax.set_xlabel("Number of Distinct Notes", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Histogram of Distinct Notes in Each Melody", fontsize=12)

    # Set x-ticks
    x_ticks = range(0, max(distinct_counts) + 1, x_tick_interval)
    ax.set_xticks(x_ticks)

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def plot_melody_length_histogram(
    melody_lengths, max_length=500, bin_width=10, figsize=(10, 6)
):
    """
    Create a histogram of melody lengths.

    Parameters:
    melody_lengths (array-like): List or array of melody lengths.
    max_length (int): Maximum length to consider for the histogram. Default is 500.
    bin_width (int): Width of each bin in the histogram. Default is 10.
    figsize (tuple): Figure size (width, height) in inches. Default is (10, 6).

    Returns:
    fig, ax: The created figure and axes objects.
    """
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Filter melody lengths to consider only those up to max_length
    filtered_lengths = [length for length in melody_lengths if length <= max_length]

    # Create bins
    bins = range(1, max_length + bin_width, bin_width)

    # Plot the histogram
    ax.hist(filtered_lengths, bins=bins, edgecolor="black", align="left")

    # Set labels and title
    ax.set_xlabel("Melody Length", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Histogram of Each Melody's Length", fontsize=12)

    # Set x-ticks (you can uncomment and modify this if needed)
    # x_ticks = range(1, max_length+1, 50)
    # ax.set_xticks(x_ticks)

    # Adjust layout
    plt.tight_layout()

    return fig, ax
