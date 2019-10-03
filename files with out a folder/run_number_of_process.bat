echo off
FOR /L %%A IN (1,1,50) DO (
	rem start C:\Users\Bar\Miniconda3\envs\thesis_env\python run_number_of_process2.py
	call C:\Users\Bar\Miniconda3\envs\thesis_env\python run_number_of_process.py

)
pause