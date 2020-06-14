from GVHD_BAR.gvhd_analysis_tf import main
import sys
import os

grid_results_folder = sys.argv[1]
use_similiarity = bool(int(sys.argv[2]))
use_censored = bool(int(sys.argv[3]))

gpu_number = str(int(sys.argv[4]))
print(f'Use censored={use_censored}, Use similarity={use_similiarity}, grid folder={grid_results_folder}')


os.environ["CUDA_VISIBLE_DEVICES"]=gpu_number
main(use_censored, use_similiarity, grid_results_folder)