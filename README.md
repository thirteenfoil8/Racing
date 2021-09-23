# Documentation 
Here is the Open Source Environment:
https://gym.openai.com/envs/CarRacing-v0/

# Setup 
Download the environment using anaconda :
cd ProjectDirectory 
conda env create -f environment.yml
conda activate logitech

then :
# Train
Carefull, the train takes a lot of time to compute the weight of the neurones. 
Run:
python .\train.py

# Test
Run:
python .\test.py

# Video of the lap
./recording/vid2.mp4