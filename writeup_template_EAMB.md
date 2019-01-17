# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/architecture.jpg "Model Visualization"
[image2]: ./examples/eaxmple_image_forward_correct_way.jpg "Correct way image example"
[image3]: ./examples/example_image_wrong_way.jpg "Wrong way image example"
[image4]: ./examples/eaxmple_cropped_image.jpg "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report_EAMB.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I used consists of 9 layers, including a normalization layer, 5 convolutional layers, 1 dropout layer and 3 fully connected layers. Which is very similar to the one used by NVIDIA for Self driving cars. Here is the [link](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) to the document where I found the arquitecture. All this arquitecture is implemented in `model.py` code lines **68 to 89**.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 85).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 118-122). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 119).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of a 2 lap data set for driving forward (normal way track) and another set of driving clockwise the track just 1 lap.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was first try the available models teached in class. That is why I tried first the GoogleNet (InceptionV3) dropping the last 2 layers and adding a a new fully connected layer. First I tried by leaving the actual values of the hypperparameters of **ImageNet** ann start the trainning from there the results of the **MSE** for the **training** and **validation** set was around **1.5** and **2** respectively. Then I tried to put random values in all the hyper parameters of all layers and train them from scratch but the result it the same. I didn't remove this architecture it is in lines 91 to 116 from `model.py`, so if you want to try is you need to modified line 10:

```python
arquitecture = 2
```
Then I change to use a similar architecture to NVIDIA architecture for Self driving cars which works fine since the beginning. The car travel along all the track without dropping to the water or going outside the track. I got results for the **training** and **validation** set around **0.016** and **0.019** respectively.

To combat the overfitting, I modified the model so that by adding a dropout layer, first I added just after all the convolutional layers, even the results were not bad around **0.018** and **0.021** the car dropped to the water. That why I changed it just after the Flatten operation, this time the results were around **0.016** for training set and **0.02** for the validation set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The video of this is in the following [link](https://github.com/emoyers/CarND-Behavioral-Cloning-P3/blob/master/final_run.mp4).

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-89) consisted of a :

* Normalizing layer.
* Cropping layer.
* Convolutional layer of 24 filters, a kernel of 5x5, strides of 2x2 and an activation **relu** function.   
* Convolutional layer of 36 filters, a kernel of 5x5, strides of 2x2 and an activation **relu** function.
* Convolutional layer of 48 filters, a kernel of 5x5, strides of 2x2 and an activation **relu** function.
* 2 Convolutional layers of 64 filters, a kernel of 3x3 and strides of 1x1.
* Flatten operation.
* Dropout function.
* 4 Fully connected layers.

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To augment the data sat, I drove clockwise (wrong way) the track for one lap.

![alt text][image3]


After the collection process, I had 5341 number of data points. I then preprocessed this data by normalizing it, performing the operation `(x/255) - 0.5`, which make the mean been around 0 that works better with tensor flow. And also I cropped the image to remove the upper part of the picture that includes the mountains and trees and also the bottom part to remove the hood of the car. The following image shows and example of the cropped image.

![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the log created by the python script `model.py`. I used an adam optimizer so that manually training the learning rate wasn't necessary.
