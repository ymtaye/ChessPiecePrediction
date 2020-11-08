# ChessPiecePrediction
Implemented Machine Learning and Convolutional Neural Networks to predict chess pieces. 

## Overview
In this project we used the chessman image dataset from kaggle.com. The
dataset can be found here: [Dataset](https://www.kaggle.com/niteshfre/chessman-image-dataset/data) \
We wanted to use this data set to train and test different AI models and Techniques that result in
good prediction accuracy, and were able to be used to predict future images and classify them
into various chess piece categories.

## CNN 
As one approach to classifying chess images, we used Convolutional Neural Networks
(CNN) with transfer learning. One of the major problems with image classification is the large
number of features that come with it. Each pixel of an image represents a feature. For a small
600x600 pixel RGB image, there are 1,080,000 features. This makes feeding such features to
neural networks impractical. CNNs are useful in extracting the useful parts of the image,
effectively shrinking down the number of features to be fed into the neural network. We can also
shrink down the size of the image before processing with CNNs. 

## My Contribution


## PCA
We implemented principal component analysis. Due to the
high dimensionality nature of our data, we needed to try a dimension reduction approach and
combine it with other prediction models. Based on our research of multiple [Dimensionality Reduction Techniques Python](https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/
) & [PCA guide](https://www.analyticsvidhya.com/blog/2016/03/pca-practical-guide-principal-component-analysis
) we decided to use Principal Component Analysis (PCA) to reduce dimensionality of
our data.

we also used a built in function to normalize the data, which is usually
recommended/required when running a PCA analysis: we used the normalize implementation
within sklearn.preprocessing library In addition, we did our dimensionality reduction using both
grayscale and original colors. [Preprocessing-images-dim-redux](https://www.kaggle.com/hamishdickson/preprocessing-images-with-dimensionality-reduction)
was used as a reference point when devising our approach.
PCA was fit on the training data set which was predetermined using a 80:20 train/test
ratio. We then attempted to find the principal components that captured 95% variance of the
dataset. We then transformed our training feature dataset (train_X) and testing X feature
dataset (test_X) using the determined components.

#### Example
<a href="url"><img src="https://github.com/ymtaye/ChessPiecePrediction/blob/main/sample/pca reduced test.png" align="left" height="200" width="200" ></a>
<a href="url"><img src="https://github.com/ymtaye/ChessPiecePrediction/blob/main/sample/PCA_sample2.png" align="center" height="200" width="200" ></a>
<a href="url"><img src="https://github.com/ymtaye/ChessPiecePrediction/blob/main/sample/Image_4.png" align="right" height="200" width="200" ></a>


## Machine Learning 



## Project Takeaways
Based on the approaches we explore in this project, we think CNN are better at
accurately classifying chess pieces compared to using a Principle Component Analysis approach.
Other things we wanted to consider for the future to further test and/or improve
performance would be:
- Expand dataset to include non chess images
- Train CNN model on images both with and without static noises
- Try other dimensionality techniques other than PCA. Some ideas would
Non-negative Matrix Factorization and Chi^2 dimension reductionality methods
- In case of PCA approach, resizing the image shapes in a different way to feed
PCA and machine learning methods. Currently converting images into numpy
array which had initial shape of (samples, width, height, RGB) to (samples,
width x height x RGB) array.
- Use a gridsearch to tune model hyper-parameters more effectively for our
dataset.
- Experiment with XGBoost models
- Overall, the PCA approach didn’t result in favourable outcomes, but has room to
be improved. Nevertheless, it is proven to be better than random prediction since
for our case random would ~ 16.67%.