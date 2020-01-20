from copy import deepcopy

from fst_tools import FSTools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class RandomFstStats:
    # if num_accept_states is --not-- None than accept_percent is dismissed
    def __init__(self, min_alphabet, max_alphabet, min_states, max_states, accept_percent=10, num_accept_states=None
                 , samples_to_check=100):
        self._min_alphabet = min_alphabet
        self._max_alphabet = max_alphabet
        self._min_states = min_states
        self._max_states = max_states
        self._total_samples = samples_to_check
        # list of list fst_matrix = [alphabet_size][state_size] -> list of k random fst
        self._random_fst = self._rand_fst_set(accept_percent, num_accept_states)

    def _heat_matrix(self, matrix, title, norm=False):
        m = deepcopy(matrix)
        max_val = np.max(np.matrix(matrix).flatten())
        min_val = np.min(np.matrix(matrix).flatten())
        for i, symbol in enumerate(range(self._min_alphabet, self._max_alphabet)):  # y_axis
            for j, state in enumerate(range(self._min_states, self._max_states)):  # x_axis
                m[i][j] = min(matrix[i][j] / max_val, 0.95) + 0.05 if norm else matrix[i][j]
        ax = sns.heatmap(m, cmap="YlGnBu",
                         yticklabels=[str(x) for x in range(self._min_alphabet, self._max_alphabet)],
                         xticklabels=[str(x) for x in range(self._min_states, self._max_states)],
                         vmin=0 if norm else min_val, vmax=1 if norm else max_val)
        ax.set_title(title)
        plt.xlabel("num states")
        plt.ylabel("num alphabet")
        plt.show()

    def plot_effective_deg(self):
        self._heat_matrix(self._effective_deg_heatmap(), "effective degree")

    def plot_sequence_len(self):
        mean_len, std_len = self._sequence_length_heatmap()
        self._heat_matrix(mean_len, "mean sequence length")
        self._heat_matrix(std_len, "std sequence length")

    def plot_accept_percent(self):
        self._heat_matrix(self._accept_percentage_heatmap(), "accept percentage")

    def _state_idx(self, idx):
        return idx - self._min_states

    def _alphabet_idx(self, idx):
        return idx - self._min_alphabet

    def _rand_fst_set(self, accept_percent, num_accept_states):
        print("\nstart randomize FSTs")
        all, curr = (self._max_alphabet - self._min_alphabet) * (self._max_states - self._min_states), 0
        fst_matrix = []
        for i, alphabet_size in enumerate(range(self._min_alphabet, self._max_alphabet)):
            fst_matrix.append([])
            for state_size in range(self._min_states, self._max_states):
                curr += 1
                print("\r\r\r\r\r\r\r\r\r\r\r" + str(int(100 * (curr / all))) + "%", end="")

                accept_size = max(1, accept_percent * state_size // 100)
                accept_size = accept_size if num_accept_states is None else num_accept_states
                sample_list = []
                for _ in range(self._total_samples):
                    rand_fst = FSTools().rand_fst(state_size, alphabet_size, accept_size)
                    while rand_fst.state("q0").is_reject:
                        rand_fst = FSTools().rand_fst(state_size, alphabet_size, accept_size)
                    sample_list.append(rand_fst)
                fst_matrix[i].append(sample_list)
        return fst_matrix

    def _effective_deg_heatmap(self):
        print("\nstart effective deg")
        all, i = (self._max_alphabet - self._min_alphabet) * (self._max_states - self._min_states), 0
        deg_matrix = []
        for num_alph in range(self._min_alphabet, self._max_alphabet):
            deg_matrix.append([])
            for num_state in range(self._min_states, self._max_states):
                i += 1
                print("\r\r\r\r\r\r\r\r\r\r\r" + str(int(100 * (i / all))) + "%", end="")

                alph_idx, state_idx = self._alphabet_idx(num_alph), self._state_idx(num_state)

                effective_deg_list = []
                for rand_fst in self._random_fst[alph_idx][state_idx]:
                    effective_deg_list.append(np.mean(list(rand_fst.effective_deg().values())))

                deg_matrix[alph_idx].append(np.mean(effective_deg_list))
        return deg_matrix

    def _sequence_length_heatmap(self):
        print("\nstart sequence length")
        all, i = (self._max_alphabet - self._min_alphabet) * (self._max_states - self._min_states), 0
        mean_matrix, std_matrix = [], []
        for num_alph in range(self._min_alphabet, self._max_alphabet):
            mean_matrix.append([])
            std_matrix.append([])
            for num_state in range(self._min_states, self._max_states):
                i += 1
                print("\r\r\r\r\r\r\r\r\r\r\r" + str(int(100 * (i / all))) + "%", end="")

                alph_idx, state_idx = self._alphabet_idx(num_alph), self._state_idx(num_state)

                sequence_len_mean_list = []
                sequence_len_std_list = []
                for rand_fst in self._random_fst[alph_idx][state_idx]:
                    fst_mean, fst_std = rand_fst.mean_variance_sequence_len()
                    sequence_len_mean_list.append(fst_mean)
                    sequence_len_std_list.append(fst_std)

                mean_matrix[alph_idx].append(np.mean(sequence_len_mean_list))
                std_matrix[alph_idx].append(np.std(sequence_len_std_list))
        return mean_matrix, std_matrix

    def _accept_percentage_heatmap(self):
        print("\nstart accept percent")
        all, i = (self._max_alphabet - self._min_alphabet) * (self._max_states - self._min_states), 0
        accept_matrix = []
        for num_alph in range(self._min_alphabet, self._max_alphabet):
            accept_matrix.append([])
            for num_state in range(self._min_states, self._max_states):
                i += 1
                print("\r\r\r\r\r\r\r\r\r\r\r" + str(int(100 * (i / all))) + "%", end="")
                alph_idx, state_idx = self._alphabet_idx(num_alph), self._state_idx(num_state)

                accept_percent_list = []
                for rand_fst in self._random_fst[alph_idx][state_idx]:
                    accept_percent_list.append(rand_fst.accept_percentage())

                accept_matrix[alph_idx].append(np.mean(accept_percent_list))
        return accept_matrix


if __name__ == "__main__":
    r = RandomFstStats(min_alphabet=5, max_alphabet=11, min_states=15, max_states=21, samples_to_check=50, num_accept_states=2)
    r.plot_effective_deg()
    r.plot_accept_percent()
    r.plot_sequence_len()
