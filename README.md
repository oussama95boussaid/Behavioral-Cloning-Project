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

# How to Run the Model

The neural network model needed to be built into the python program file **«model.py»**. When this program was launched; needed to save the model to the **«model.h5»** file. This saved neural network model can then be invoked to run Car Simulator by running the command **«python drive.py model.h5»**. A helper file **«drive.py»** was provided by Udacity that we could modify; if we needed.


For data collection; the following procedure must be followed: When the car drives through the simulator in «Training» mode; it saves the images in **the IMG folder** and the details of the images are stored in **the driving_log.csv file**. The drving_log.csv file has 7 columns; in the following order: Middle camera image file name, Left camera image file name, Right camera image file name, Steering , Throttle , Brake and Speed. Build a model; we could use either only the image from the central camera or from all three cameras. In the output, only the Steering value was relevant for this project; Throttle, brake and speed values were ignored.

The file **«video.py»** was created from the dataset using the command **«python video.py run1»**; where «run1» is the folder where the dataset is stored.

**Why create a video**

- It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
- You could slightly alter the code in drive.py and/or video.py to create a video of what your model sees after the image is processed (may be helpful for debugging).

# Simulator Download

- <a href="https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip">Linux</a>
- <a href="https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip">Mac</a>
- <a href="https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip">Windows</a>


During training, the human demonstrator drives the vehicle using his/her keyboard, as in a video game:

<img src ="img/train_screen.png" >

When you first run the simulator, you’ll see a configuration screen asking what size and graphical quality you would like. We suggest running at the smallest size and the fastest graphical quality.
We also suggest closing most other applications (especially graphically intensive applications) on your computer, so that your machine can devote its resources to running the simulator.

*Collecting Training Data*

In order to start collecting training data, you'll need to do the following:

1. Enter Training Mode in the simulator.
2. Start driving the car to get a feel for the controls.
3. When you are ready, hit the record button in the top right to start recording.
4. Continue driving for a few laps or till you feel like you have enough data.
5. Hit the record button in the top right again to stop recording.

*Strategies for Collecting Data*

- the car should stay in the center of the road as much as possible
- if the car veers off to the side, it should recover back to center
- driving counter-clockwise can help the model generalize
- flipping the images is a quick way to augment the data
- collecting data from the second track can also help generalize the model
- we want to avoid overfitting or underfitting when training the model
- knowing when to stop collecting more data
