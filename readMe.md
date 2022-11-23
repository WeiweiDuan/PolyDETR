# PolyDETR: Polyline Detection Transformer

## PolyDETR's goal is to detect polylines' location in the topographic maps

## Docker imagery to train/testing PolyDETR
**Here is the command to run the docker imagery**

<code>sudo nvidia-docker run -t -i -v {local_dir}:{docker_dir} -p 8888:8888 pytorch/pytorch:1.2-cuda10.0-cudnn7-devel</code>

## Input topographic maps
The topographic maps from the map competition. The competition provides tif map images, and tif label images. Label images are binary {0, 1}, 1 reprensents the pixels belong to the desired polyline feature.

## Train PolyDETR
The training process includes 
1) data processing: convert tif map images to png images, and tif label image to vector (shapefile). The png map images and vector polylines are training data
2) data augmentation: includes shifting along x- and y-axis, and rotation every 90 degrees.
3) training PolyDETR

To update the parameters for data processing, model architecture, and training process, please update './util/args.py'

**Here is the command to train PolyDETR
<code> python train.py </code>

## Use PolyDETR to detect desired polylines
Update './util/args_test.py' to set the map name and trained model path for PolyDETR

## Here is the command to test PolyDETR
<code> python test.py </code>

## Here is the command to use PolyDETR to detect line features for the competition
<code> python test_main_competition.py </code>
### Please update './util/args_test_multimaps.py' to set directory for the map images 


