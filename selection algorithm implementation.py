import random
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

class SelectionAlgorithms:
    """
    Implementation of deterministic and randomized selection algorithms
    for finding the k-th smallest element in an array.
    """
    
    def __init__(self):
        self.deterministic_comparisons = 0
        self.randomized_comparisons = 0
    
    # ==================== DETERMINISTIC SELECTION (Median of Medians) ====================
    
    def deterministic_select(self, arr: List[int], k: int) -> int:
        """
        Deterministic selection algorithm using Median of Medians.
        Finds the k-th smallest element (1-indexed) in O(n) worst-case time.
        
        Args:
            arr: Input array
            k: The order statistic to find (1-indexed)
        
        Returns:
            The k-th smallest element
        """
        if not arr or k < 1 or k > len(arr):
            raise ValueError("Invalid input")
        
        self.deterministic_comparisons = 0
        return self._deterministic_select_helper(arr.copy(), 0, len(arr) - 1, k - 1)
    
    def _deterministic_select_helper(self, arr: List[int], left: int, right: int, k: int) -> int:
        """
        Helper function for deterministic selection.
        
        Args:
            arr: Array to search in
            left: Left boundary
            right: Right boundary
            k: Index of the element to find (0-indexed)
        """
        if left == right:
            return arr[left]
        
        # Find median of medians as pivot
        pivot_value = self._median_of_medians(arr, left, right)
        
        # Partition around the pivot
        pivot_index = self._partition_deterministic(arr, left, right, pivot_value)
        
        # Recursively select
        if k == pivot_index:
            return arr[k]
        elif k < pivot_index:
            return self._deterministic_select_helper(arr, left, pivot_index - 1, k)
        else:
            return self._deterministic_select_helper(arr, pivot_index + 1, right, k)
    
    def _median_of_medians(self, arr: List[int], left: int, right: int) -> int:
        """
        Find the median of medians to use as a pivot.
        Divides array into groups of 5, finds median of each, then finds median of medians.
        """
        n = right - left + 1
        
        if n <= 5:
            return self._find_median(arr, left, right)
        
        # Divide into groups of 5 and find median of each group
        medians = []
        for i in range(left, right + 1, 5):
            group_right = min(i + 4, right)
            median = self._find_median(arr, i, group_right)
            medians.append(median)
        
        # Find median of medians recursively
        return self._deterministic_select_helper(medians, 0, len(medians) - 1, len(medians) // 2)
    
    def _find_median(self, arr: List[int], left: int, right: int) -> int:
        """Find median of a small subarray using insertion sort."""
        # Extract subarray
        subarr = arr[left:right + 1]
        subarr.sort()
        self.deterministic_comparisons += len(subarr) * len(subarr)  # Approximate
        return subarr[len(subarr) // 2]
    
    def _partition_deterministic(self, arr: List[int], left: int, right: int, pivot_value: int) -> int:
        """
        Partition array around pivot value.
        Returns the final position of the pivot.
        """
        # Find the pivot index
        pivot_index = left
        for i in range(left, right + 1):
            if arr[i] == pivot_value:
                pivot_index = i
                break
        
        # Move pivot to end
        arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
        
        # Standard partition
        store_index = left
        for i in range(left, right):
            self.deterministic_comparisons += 1
            if arr[i] < pivot_value:
                arr[i], arr[store_index] = arr[store_index], arr[i]
                store_index += 1
        
        # Move pivot to final position
        arr[store_index], arr[right] = arr[right], arr[store_index]
        return store_index
    
    # ==================== RANDOMIZED SELECTION (Randomized Quickselect) ====================
    
    def randomized_select(self, arr: List[int], k: int) -> int:
        """
        Randomized selection algorithm (Quickselect with random pivot).
        Finds the k-th smallest element (1-indexed) in O(n) expected time.
        
        Args:
            arr: Input array
            k: The order statistic to find (1-indexed)
        
        Returns:
            The k-th smallest element
        """
        if not arr or k < 1 or k > len(arr):
            raise ValueError("Invalid input")
        
        self.randomized_comparisons = 0
        return self._randomized_select_helper(arr.copy(), 0, len(arr) - 1, k - 1)
    
    def _randomized_select_helper(self, arr: List[int], left: int, right: int, k: int) -> int:
        """
        Helper function for randomized selection.
        
        Args:
            arr: Array to search in
            left: Left boundary
            right: Right boundary
            k: Index of the element to find (0-indexed)
        """
        if left == right:
            return arr[left]
        
        # Randomly select pivot and partition
        pivot_index = self._randomized_partition(arr, left, right)
        
        # Recursively select
        if k == pivot_index:
            return arr[k]
        elif k < pivot_index:
            return self._randomized_select_helper(arr, left, pivot_index - 1, k)
        else:
            return self._randomized_select_helper(arr, pivot_index + 1, right, k)
    
    def _randomized_partition(self, arr: List[int], left: int, right: int) -> int:
        """
        Partition with random pivot selection.
        Returns the final position of the pivot.
        """
        # Choose random pivot
        pivot_idx = random.randint(left, right)
        pivot_value = arr[pivot_idx]
        
        # Move pivot to end
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        
        # Partition
        store_index = left
        for i in range(left, right):
            self.randomized_comparisons += 1
            if arr[i] < pivot_value:
                arr[i], arr[store_index] = arr[store_index], arr[i]
                store_index += 1
        
        # Move pivot to final position
        arr[store_index], arr[right] = arr[right], arr[store_index]
        return store_index


# ==================== EMPIRICAL ANALYSIS ====================

def generate_test_data(size: int, data_type: str) -> List[int]:
    """Generate test data of different types."""
    if data_type == "random":
        return [random.randint(1, size * 10) for _ in range(size)]
    elif data_type == "sorted":
        return list(range(1, size + 1))
    elif data_type == "reverse":
        return list(range(size, 0, -1))
    elif data_type == "duplicates":
        return [random.randint(1, size // 10) for _ in range(size)]
    elif data_type == "nearly_sorted":
        arr = list(range(1, size + 1))
        # Swap 10% of elements
        for _ in range(size // 10):
            i, j = random.randint(0, size - 1), random.randint(0, size - 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    return []


def benchmark_algorithms():
    """Empirical comparison of both algorithms."""
    selector = SelectionAlgorithms()
    sizes = [100, 500, 1000, 2500, 5000, 7500, 10000]
    data_types = ["random", "sorted", "reverse", "duplicates", "nearly_sorted"]
    
    results = {
        "deterministic": {dt: [] for dt in data_types},
        "randomized": {dt: [] for dt in data_types}
    }
    
    print("=" * 80)
    print("EMPIRICAL ANALYSIS OF SELECTION ALGORITHMS")
    print("=" * 80)
    
    for data_type in data_types:
        print(f"\n{data_type.upper()} DATA:")
        print("-" * 80)
        print(f"{'Size':<10} {'Det Time (ms)':<15} {'Det Comps':<15} {'Rand Time (ms)':<15} {'Rand Comps':<15}")
        print("-" * 80)
        
        for size in sizes:
            # Generate test data
            arr = generate_test_data(size, data_type)
            k = size // 2  # Find median
            
            # Test deterministic algorithm
            start = time.perf_counter()
            result_det = selector.deterministic_select(arr, k)
            det_time = (time.perf_counter() - start) * 1000
            det_comps = selector.deterministic_comparisons
            
            # Test randomized algorithm (average over 5 runs)
            rand_times = []
            rand_comps_list = []
            for _ in range(5):
                start = time.perf_counter()
                result_rand = selector.randomized_select(arr, k)
                rand_times.append((time.perf_counter() - start) * 1000)
                rand_comps_list.append(selector.randomized_comparisons)
            
            rand_time = np.mean(rand_times)
            rand_comps = np.mean(rand_comps_list)
            
            # Verify correctness
            assert result_det == result_rand == sorted(arr)[k - 1]
            
            results["deterministic"][data_type].append(det_time)
            results["randomized"][data_type].append(rand_time)
            
            print(f"{size:<10} {det_time:<15.3f} {det_comps:<15} {rand_time:<15.3f} {rand_comps:<15.1f}")
    
    return sizes, results


def plot_results(sizes: List[int], results: dict):
    """Plot performance comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Selection Algorithms Performance Comparison', fontsize=16)
    
    data_types = ["random", "sorted", "reverse", "duplicates", "nearly_sorted"]
    
    for idx, data_type in enumerate(data_types):
        ax = axes[idx // 3, idx % 3]
        
        det_times = results["deterministic"][data_type]
        rand_times = results["randomized"][data_type]
        
        ax.plot(sizes, det_times, 'b-o', label='Deterministic', linewidth=2)
        ax.plot(sizes, rand_times, 'r-s', label='Randomized', linewidth=2)
        ax.set_xlabel('Input Size')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{data_type.replace("_", " ").title()} Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('selection_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    print("\n\nPlot saved as 'selection_algorithms_comparison.png'")
    plt.show()


# ==================== DEMONSTRATION ====================

if __name__ == "__main__":
    # Simple demonstration
    print("\n" + "=" * 80)
    print("SIMPLE DEMONSTRATION")
    print("=" * 80)
    
    selector = SelectionAlgorithms()
    test_arr = [3, 7, 2, 9, 1, 5, 8, 4, 6]
    print(f"\nArray: {test_arr}")
    print(f"Sorted: {sorted(test_arr)}")
    
    for k in [1, 3, 5, 7, 9]:
        det_result = selector.deterministic_select(test_arr, k)
        rand_result = selector.randomized_select(test_arr, k)
        expected = sorted(test_arr)[k - 1]
        
        print(f"\nk={k}: Deterministic={det_result}, Randomized={rand_result}, Expected={expected}")
        assert det_result == rand_result == expected
    
    print("\n✓ All tests passed!")
    
    # Run empirical analysis
    print("\n\nRunning comprehensive empirical analysis...")
    sizes, results = benchmark_algorithms()
    
    # Plot results
    plot_results(sizes, results)