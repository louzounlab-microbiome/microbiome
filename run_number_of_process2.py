from allergy.analysis_using_tf import main
import sys

grid_results_folder = sys.argv[1]
use_similiarity = bool(int(sys.argv[2]))
use_censored = bool(int(sys.argv[3]))

print(f'Use censored={use_censored}, Use similarity={use_similiarity}, grid folder={grid_results_folder}')
main(use_censored, use_similiarity, grid_results_folder)