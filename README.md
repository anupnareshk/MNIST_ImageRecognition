# MNIST Hand written Image Recognition

### Problem statement:
Original dataset consists of 42000 images of hand written digits. So based on this dataset classification model has to be generated to identify the digits. So when an pixel values of image is given model should identify the digit in the image.

</p>
</details>
<p align="center"> 
<img src="https://github.com/anupnareshk/MNIST_ImageRecognition/blob/master/Images/test.png" width="600" height="400">
</p>

#### Exploratory data analysis:
Training dataset consists of 42000 images which is 28 X 28 pixels of gray scale images. Those 28 X 28 pixels are converted to vectors, so data set consists of 784 columns along with ‘label’ .  The pixel value ranges from 0 – 255. 

Before jumping into model development we need to visualize the data set. Since the data is multi-dimensional we will use PCA to reduce it to 2 dimensions for visualization.

* Principal Component Analysis (PCA)
Principal Component Analysis shows clear difference between the images ranging from 0-9.
</p>
</details>
<p align="center"> 
<img src="https://github.com/anupnareshk/MNIST_ImageRecognition/blob/master/Images/2PCA.png" width="600" height="400">
</p>

####	Data Preprocessing:
Since training dataset consists of 784 columns it is better to reduce the dimensionality of the dataset using PCA. 100 principal components are generated for faster computation and analysis. 

####	Model development:
KNN and SVM classifier models are developed, so based on the accuracy score the best model is selected. 

####	Interpretation of Results:
Model results are in the form of probability scores, based on the scores we can decide which users are interested in purchasing the product and target those users. This score can be used to segment into different classes based on there website section or product visited. If we segment into different classes, then this will also tell which user has spent time on browsing but not purchased any product, so seller should target those users by providing some offers. This score can be effectively used to build strategies for selling the product and results improving business. 

###	Conclusion:
The trasnformed dataset with 100 principal components are used to train the KNN model. 
Upon calculating the accuracy of the trained model with the validation dataset. KNN has around 97% accuracy and SVM has around 98% accuracy. 

###	Visualization:
Visualization of first 40 images with predicted values
</p>
</details>
<p align="center"> 
<img src="https://github.com/anupnareshk/MNIST_ImageRecognition/blob/master/Images/KNNTestPred.png" width="600" height="400">
</p>

### Scope of improvement:
* Better analysis for choosing the number of Principal Components can be done to imporve the accuracy.
* Using Deep Learning models can lead to better accuracy of the models.
