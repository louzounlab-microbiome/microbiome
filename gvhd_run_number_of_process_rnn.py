from GVHD_BAR.gvhd_analysis_rnn import main
import sys
import os

grid_results_folder = sys.argv[1]
use_similiarity = bool(int(sys.argv[2]))
use_censored = bool(int(sys.argv[3]))

print(f'Use censored={use_censored}, Use similarity={use_similiarity}, grid folder={grid_results_folder}')


numbers_of_gpu = 4

for gpu_num in range(numbers_of_gpu):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)
        main(use_censored, use_similiarity, grid_results_folder)
        break
    except:
        pass