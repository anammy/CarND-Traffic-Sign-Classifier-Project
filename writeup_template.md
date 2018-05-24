# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/visualexp1.jpg "Visualization"
[image2]: ./figures/gray.jpg "Grayscaling"
[image3]: ./figures/normalize.jpg "Normalization"
[image4]: ./figures/rotated.jpg "Rotated"
[image5]: ./figures/translated.jpg "Translated"
[image6]: ./figures/zoomed.png "Zoomed"
[image7]: ./figures/augment.jpg "Augment"
[image8]: ./figures/newimages.jpg "NewImg"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. PUT MY GITHUB LINK HERE!
The following is the report on the Traffic Sign Classifier project (See [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb))

### Data Set Summary & Exploration

#### 1. Dataset Summary

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory Visualization of the Dataset.

The following histogram displays the distribution of the classes within the training, validation, and testing data sets.

![alt text][image1]

The following graphs display the distribution of classes in the training, validation, and testing sets. The following classes occur at a significantly higher frequency:

* 1,Speed limit (30km/h)
* 2,Speed limit (50km/h)
* 4,Speed limit (70km/h)
* 5,Speed limit (80km/h)
* 10,No passing for vehicles over 3.5 metric tons
* 12,Priority road
* 13,Yield
* 38,Keep right

From the normed histogram in the project code, the distribution of classes are nearly identical between the training, validation, and testing sets.

### Design and Test a Model Architecture

#### 1. Preprocessing

The images were first converted to grayscale and then, normalized using the equation (pixel - mean)/(standard deviation). This yielded a more uniform normalization with a zero mean and equal variance than using the equation (pixel - 128)/128.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Here is an example of a traffic sign image before and after normalization.

![alt text][image3]

I decided to generate additional data to augment the training set and avoid bias in the model since the training set distribution was skewed towards a few classes.

To add more data to the the data set, images from low frequency classes in the training set were rotated, translated, and zoomed to varying degrees. The aim of the augmentation was to have a minimum of 750 images per class in the training set.

Here is an example of an original image and an augmented image:

![alt text][image4]

The following histogram shows the classes which were augmented:

![alt text][image7]

#### 2. Model Architecture

My final model was a variation of the LeNet model (from the previous assignment) and consisted of the following layers:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|         										|
| Max pooling			| 2x2 stride,  outputs 5x5x6        			|
| Fully Connected		| Input = 400, Output = 120						|
| RELU					|												|
| Dropout				| 												| 
| Fully Connected		| Input = 120, Output = 84						|
| RELU					|												|
| Dropout				| 												|
| Fully Connected		| Input = 84, Output = 43						|

The dropout layers were added in order to avoid overfitting the model. Adding the dropout layers to the LeNet architecture with a probability of keeping values of 0.5 increased the validation accuracy by approximately 2%.

#### 3. Model Training

To train the model, I used the Adam Optimizer function since it was already implemented and worked well in the LeNet lab. I varied parameters such as the batch size between 100 to 200, number of epoches between 30 to 150, and the learning rate between 0.005 to 0.0001.

#### 4. Solution Approach

I implemented the LeNet architecture since it worked well for the analysis of 32x32x1 MNIST images in the previous assignment. I ran this model with some variations in the hyperparameters to get an idea of its effect on the validation accuracy. I also alternated between max pooling and average pooling, which didn't have a significant impact on the validation accuracy. The LeNet architecture was then modified to include dropout layers to prevent overfitting. The accuracy improved by approx. 2%. I then varied parameters such as the batch size, learning rate, epochs, and the probability of keeping values in the dropout layer (i.e. keep_prob) to see its effect on the validation accuracy. The following hyperparameters were chosen after multiple iterations:

Hyperparameters:
* Batch size: 100
* Learning rate: 0.0009
* Epochs: 75
* Keep_prob: 0.5

I also augmented the training data set by flipping, rotating, translating and zooming images since it was skewed towards a few classes. At first, this seemed to reduce the validation accuracy. I then lowered the rotation angle and removed the flip operation in the augmented data set. Image operations with large, agressive changes were reducing the accuracy once the training set was augmented. Validation accuracy improved by approx. 1%.

My final model results were:
* Training set accuracy of 99.8%
* Validation set accuracy of 96.8% 
* Test set accuracy of 95%

This model yielded a training, validation, and test accuracy greater than the project criteria of 93% and thus, was judged to perform well.

I also modified the LeNet architecture by removing one of the fully connected layers and widening the second last fully connected layer so that the input from 800 nodes would be processed into the 43 classes at the output layer. I was curious about the performance of wider vs. deeper networks. This wider network yielded similar results to the modified LeNet architecture above. I also tried the tanh activation function since it keeps the negative values from the convolution layer and rescales the data between -1 to 1, centered at zero. In addition, I tried both max and average pooling. These didn't have a significant impact on the validation accuracy after varying the hyperparameters. Then I tried to remove the pooling layers and found that the accuracy reduces due to overfitting. These modifications were made to further my understanding of the effect of model architecture on the validation accuracy.


### Test the Model on New Images

#### 1. New Images

Here are ten German traffic signs that I found on the web:

![alt text][image8]

I don't think these images should be difficult to classify since they look similar to the images from the training set after being rescaled to the same size.

#### 2. Performance on New Images

Here are the results of the prediction:

| Image			        				|     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)		      		| Speed limit (30km/h)							| 
| No entry				     			| No entry 										|
| Right-of-way at the next intersection	| Right-of-way at the next intersection			|
| Children crossing	      				| Children crossing				 				|
| Stop									| Stop			      							|
| Slippery road							| Slippery Road      							|
| Road work								| Bicycles crossing    							|
| No vehicles							| No vehicles	      							|
| Priority road							| Priority road      							|
| Yield									| Turn left ahead      							|

The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. This is lower than the accuracy on the test set of 95%.

#### 3. Model Certainty-Softmax Probabilities

The following table shows the soft max probabilities and predictions outputted by the model on the new test images. The model predicts all images correctly except for 7 and 10 with a high degree of certainty. It does have some trouble distinguishing between the speed limit numeric values for image 1. It predicts with 89% probability of the image containing 30km/h speed limit vs. 11% probability of the image containing 20 km/h. This may indicate that the quality or the preprocessing of the speed limit signs were poor and thus, the model would have trouble determing the numeric value of the speed limit. For image 7, the model predicts that the image is a Bicycles Crossing sign instead of a Road Work sign. Both signs have a triangular shape, but the model had trouble distinguishing the detail in the picture inside the triangle to distinguish between the two. In the future, I would output more of the erroneous results from the test set and see if this is a consistent failure due to poor image quality or preprocessing. The model also erroneously predicts the yield sign is a Turn Left Ahead sign with 43% probability vs. correctly predicting it as a Yield sign with 35% probability. Both signs have triangular features (i.e. the yield sign shape and the head of the arrow in the left turn symbol). However, I don't understand how this error occurred since the yield sign is well-represented in the training data set and the overall shape of the two signs, a major feature, are different.

**Image 1:**

|Probabilities	|Prediction					|
|:--------------:|:--------------------------:| 
|0.89			|Speed limit (30km/h)		|
|0.11			|Speed limit (20km/h)		|
|0.00			|Speed limit (70km/h)		|
|0.00			|End of speed limit (80km/h)|
|0.00			|Speed limit (80km/h)		|


**Image 2:**

|Probabilities	|Prediction|
|:--------------:|:--------------------------:|
|1.00		|No entry|
|0.00		|Vehicles over 3.5 metric tons prohibited|
|0.00		|Stop|
|0.00		|Turn right ahead|
|0.00		|Turn left ahead|


**Image 3:**

|Probabilities	|Prediction|
|:--------------:|:--------------------------:|
|1.00		|Right-of-way at the next intersection|
|0.00		|Beware of ice/snow|
|0.00		|Pedestrians|
|0.00		|Double curve|
|0.00		|Speed limit (100km/h)|


**Image 4:**

|Probabilities	|Prediction|
|:--------------:|:--------------------------:|
|1.00		|Children crossing|
|0.00		|Dangerous curve to the right|
|0.00		|Beware of ice/snow|
|0.00		|Pedestrians|
|0.00		|Road narrows on the right|


**Image 5:**

|Probabilities	|Prediction|
|:--------------:|:--------------------------:|
|1.00		|Stop|
|0.00		|Keep right|
|0.00		|Turn left ahead|
|0.00		|No entry|
|0.00		|Turn right ahead|


**Image 6:**

|Probabilities	|Prediction|
|:--------------:|:--------------------------:|
|1.00		|Slippery road|
|0.00		|Bumpy road|
|0.00		|Bicycles crossing|
|0.00		|Wild animals crossing|
|0.00		|Dangerous curve to the right|


**Image 7:**

|Probabilities	|Prediction|
|:--------------:|:--------------------------:|
|1.00		|Bicycles crossing|
|0.00		|Road work|
|0.00		|Bumpy road|
|0.00		|Beware of ice/snow|
|0.00		|Wild animals crossing|


**Image 8:**

|Probabilities	|Prediction|
|:--------------:|:--------------------------:|
|1.00		|No vehicles|
|0.00		|Bumpy road|
|0.00		|Yield|
|0.00		|Speed limit (70km/h)|
|0.00		|Ahead only|


**Image 9:**

|Probabilities	|Prediction|
|:--------------:|:--------------------------:|
|1.00		|Priority road|
|0.00		|No vehicles|
|0.00		|Yield|
|0.00		|Roundabout mandatory|
|0.00		|Keep right|


**Image 10:**

|Probabilities	|Prediction|
|:--------------:|:--------------------------:|
|0.43		|Turn left ahead|
|0.35		|Yield|
|0.20		|Keep right|
|0.01		|Ahead only|
|0.01		|Priority road|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Visualization of the Neural Network Layers

The Speed limit (30km/h) sign from the web was inputted into the visualization function. The first convolution layer (conv1) extracts the shape of the speed sign and the numeric value of the speed limit. The next Relu layer (relu1) extracts the positive pixels from the output of conv1 and sets the negative pixels to 0. As a result, the contrast of the image has increased. The next pool layer (pool1) blurs the images. The subsequent layers further distort the image and make it unrecognizable. From the conv1 and relu1 layers, it seems like the neural network takes the shape of the sign and the numeric value inside to classify the image. I wonder if pool1 destroys some of the numeric information of the speed limit prior to inputting the images into conv2. Further investigation is required to determine if pool1 should be removed and whether this explains the difficulty the model has in determine the numeric speed limit in the sign (see Section 3 in Test the Model on New Images).

Visualizing image 10 (yield sign) from the new images obtained from the web, the model recognizes the triangular shape of the yield sign. However, it misclassified this sign as the Turn left ahead sign. The original image from the web had a high aspect ratio and was resized to 32x32x1 without cropping the image first. This distorted the aspect ratio of the triangular shape of the sign. I wonder if this aspect ratio distortion caused the model to misclassify this sign if the more slender triangular shape matched the head of the arrow in the left turn symbol. The yield sign was well represented in the training data set and thus, was not augmented through image operations. Further investigation warranted.

