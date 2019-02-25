echo off
title allergy_multi_grid_xgboost_without_similiarity

FOR /L %%B IN (1,1,5) DO (
	FOR /L %%A IN (1,1,10) DO (
	call C:\Users\Bar\Miniconda3\envs\thesis_env\python run_number_of_process.py allergy_multi_grid_xgboost_without_similiarity 0
)
)
pause