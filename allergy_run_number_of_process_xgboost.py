from allergy.analysis_using_similiarty import main
import sys

grid_results_folder = sys.argv[1]
use_similiarity = bool(int(sys.argv[2]))
print(f'Use similarity={use_similiarity}, grid folder={grid_results_folder}')
main(use_similiarity, grid_results_folder)