# Behavioral_Cloning_Project : Predicting Steering Angles from Captured By Camera 

This project performs behavioral cloning, training an AI agent to mimic human driving behavior in a simulator. Using the vehicle's camera images collected from the human demonstration, we train a deep neural network to predict the vehicle's steering angle.  

The final trained model is tested on the same test track that was run during the human demonstration.

<img src ="img/projectStructure.png" >

# Objective Of The Project

- Use the simulator to collect data of good driving behavior
- Design, train and validate a model that predicts a steering angle from image data
- Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
- Summarize the results with a written report

# Dependencies

- Python 3.10 / 3.7
- TensorFlow
- Keras
- PIL
- Numpy
- h5py
- Scikit Learn
- Pandas
- OpenCV
- Matplotlib (Optional)
- Udacity behavioral cloning simulator

# Project Stepts :

- step 1 : Collecting data ( from the human demonstration) using simulator for good driving behavior
- step 2 : Load The data
- step 3 : Split data into trainig and validation sets
- step 4 : Define a generator function to be used through training
- step 5 : Use the defined generator for training set and validation set
- step 6 :  Using keras, build a regression model based on nvidia architecture  to predict the steering angle

# My Project Includes The Following Files 

- <a href= "Behavioral_Cloning_Project.ipynb">Behavioral_Cloning_Project.ipynb</a> containing the script to create and train the model
- <a href= "drive.py"> drive.py </a>for driving the car in autonomous mode
- <a href= "model.h5">model.h5</a> containing a trained convolution neural network
- <a href= "video.py">video.py</a> a script that can be used to make a video of the vehicle when it is driving autonomously
- <a href= "README.md">README.md</a> summarizing the results
- images folder contains the sample images

# Simulator Download

- <a href="https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip">Linux</a>
- <a href="https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip">Mac</a>
- <a href="https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip">Windows</a>


During training, the human demonstrator drives the vehicle using his/her keyboard, as in a video game:

<img src ="img/train_screen.png" >
<img src = ""
