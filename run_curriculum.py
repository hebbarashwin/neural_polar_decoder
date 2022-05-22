import subprocess
  
  
# From Python3.7 you can add 
# keyword argument capture_output
print(subprocess.run(["python", "run_models.py"], capture_output=True))