                                                     Behavioral Cloning Project
The behavioral cloning in very simple terms is teaching the vehicle how to drive 
using your driving experience. The machine simply learns from the experience and 
learns to drive by itself. I believe that this will have a very strong impact on how 
the vehicle functions and it should be able to drive through the desired path or 
trajectory it has to follow given the right input data.

By using the provided simulator and the drive.py file, the car can be 
driven autonomously around the first track by simply running the following 
command
python drive.py model.h5
My model.py file contains the code for training and saving the convolution neural 
network. The file contains the methods and strategies that I used for training and 
validating the model, and it also contains comments explaining how the code 
works.

The Model Architecture and Training Strategy
My model consists of 5 convolution neural networks with first three layers having 
filters of 5x5 and two layers having filters 3x3.The number of neurons have been 
increased with each passing layer and lie between 24 and 64 in the convolutional 
network (model.py lines 70-90) The model also includes RELU layers to introduce nonlinearity in every single 
layers and the whole data is normalized in the model using the Keras lambda layer 
(code line 71). 
2. What attempts I took to reduce over fitting in the model
The model contains a L2 regularizer at every single layer which is Ridge Regression 
and also Dropouts have been used with a 50% probability of dropping neurons 
randomly. (code line 70-90)

The model was then trained on a particular set of images and validated the 
performance using another set of images to ensure that the model was not over 
fitting (code line 94-98). The model was tested by running it through the 
simulator and ensuring that the vehicle could stay on the track. A random seed 
has also been used for reproducibility so that each time we split the data, the 
same data gets assigned for training and validation hence giving us reliable 
results.

3. Model parameter tuning and the optimizer used
The model used an Adam optimizer with a learning rate of 0.0001

93. Appropriate training data
The training data was accordingly chosen to keep the vehicle driving on the road. 
The combination of center lane driving, left camera images, right 
camera images,recovering from the left and right sides of the road, images of 
particular sharp turns so that my vehicle could drive better. 

1. Solution Design Approach
The overall strategy for deriving a model architecture was to use the base of
Nvidia architecture as a starting point and then tweak 
from there. This turned out to be really helpful as I saved a lot of hours and 
possibly achieved the best results.
The first step I took to make sure that I was reading the images properly and that 
the images were getting passed to my layers was to develop a naïve model with a 
single layer. This helped me a lot in the debugging process but clearly the model 
was too simple to handle over fitting.
To combat the over fitting, I went back to my plan of using the Nvidia architecture 
and start tweaking from there. Initially, the augmented the images from the 
center camera and flipped them from left to right and vice versa. This helped me 
in more data and better model generalization. I decided to test how well the car 
was driving around track one. 
It was evident that the center camera images weren’t enough as the vehicle 
struggled to hit the sharp turns. So I decided to use the left and right camera 
images as well with a small correction factor. Finding the correction factor was 
really tricky as the vehicle would hit some turns with perfection but in some 
turns, it steered way too much. After testing a variety of values from 0.1 to 0.5, I 
figured the best value for the left images was to add 0.45 and for the right images 
to subtract 0.3. I made use of generators instead of a function and that really 
helped me speed up my training process and I could easily pass a batch of images 
now. I found a batch size of 32 seemed to perform the best.

At the end of the whole process, the vehicle is able to drive autonomously around 
the track without leaving the road.2. Final Model Architecture
The final model architecture (model.py lines 70-90) consisted of a convolution 
neural network with 5 layers as below-
 Lambda layer to normalize the data dividing by 255 and subtracting by a 
small factor epsilon(0.5 in my case)
 Cropping layer to crop 70 pixels from top and 25 pixels from the bottom.
 Convolution layer with 24 neurons, filter size of 5x5, padding set to valid, L2 
regularizer and activation relu.
 Max Pool layer with default pool size
 Convolution layer with 36 neurons, filter size of 5x5, padding set to valid, L2 
regularizer and activation relu.
 Max Pool layer with default pool size
 Convolution layer with 48 neurons, filter size of 5x5, padding set to valid, L2 
regularizer and activation relu.
 Max Pool layer with default pool size
 Convolution layer with 64 neurons, filter size of 3x3, padding set to same, 
L2 regularizer and activation relu.
 Convolution layer with 64 neurons, filter size of 3x3, padding set to valid, L2 
regularizer and activation relu.
 Max Pool layer with default pool size
 Flatten layer
 Dense layer with 80 neurons and L2 regularizer.
 Dropout layer
 Dense layer with 40 neurons and L2 regularizer.
 Dropout layer
 Dense layer with 16 neurons and L2 regularizer.
 Dropout layer
 Dense layer with 10 neurons and L2 regularizer.
 Dense layer with single neuron and L2 regularizer.

3. Creation of the Training Set & Training Process
To capture the best driving behavior, I first recorded two laps on track one using 
center lane driving. 
After the image collection process, I had 24,801 images I then preprocessed this data by 
cropping the pixels which I felt would be unnecessary like the mountains etc.(70 
pixels from the top and 25 pixels from the bottom).
I finally randomly shuffled the data set and put 20% of the data into a validation 
set. 
I used this training data for training the model. The validation set helped 
determine if the model was over or under fitting. The ideal number of epochs was 
5 and an Adam optimizer was used
