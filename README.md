# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests. (yes)
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view). (yes)
3. Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ## Reflection
   
![alt text](https://github.com/solix/CarND-Semantic-Segmentation/blob/master/runs/1509380977.0173564/um_000018.png)

The goal of the project is to manipulate vgg pretrained model and create an encoder, decoder Fully Conv neural networks as proposed in [Jonathan Long et al. paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

Schematic view of the model proposed in the paper is provided below:
![alt text](https://github.com/solix/CarND-Semantic-Segmentation/blob/master/Screenshot%202017-12-11%2013.56.40.png)

In this model first the encoding part `POOL 3` , `POOL 4` and `pool 5` layer is extracted and then decoding part is implemented as such that fully connected architecture can be build. you can find the implementation of load vgg model in `load_vgg(sess, vgg_path)` function in `main.py`.
After loading the model the fully connected convolutional layer is created in `layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)` function in `main.py`. The architecture imitates the proposed technique by [Jonathan Long et al. paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). One of the main modification in vgg architecture is upsapling the network so that we can produce pixel wise classification of the road for that reason The fully-connected layers are replaced by 1-by-1 convolutions and then we have upsample the input to the original image size. The final shape of the tensor after the final convolutional transpose layer will be 4-dimensional: `(batch_size, original_height, original_width, num_classes)`. 

Samples of the image can be found in `runs` folder.

### Next steps
   - train pedestrian segmentation and combine two models
   - create a sample test video 
