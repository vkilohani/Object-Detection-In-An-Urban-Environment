# Object Detection in an Urban Environment

## Prerequisites

`requirements.txt`

### Classroom Workspace

In the classroom workspace, every library and package should already be 
installed in your environment. You will NOT need to make use of `gcloud` to 
download the images.

## Data

The data used for this project exists in the udacity workspace in the form of 
`.tfrecord` files. These files have not been uploaded to github.

## Project Folder Structure


## Notebooks

### Exploratory Data Analysis

The `Exploratory Data Analysis` notebook has some basic data exploration. The
 most important function is `display_images`, which displays a specified 
number images with bounding boxes in a customizable grid format. The color 
code used for the bounding boxes is: red for cars, blue for pedestrians and 
green for cyclists.


### Explore augmentations

The `Explore augmentations` notebook visualizes the effect of different 
augmentations. As described later, this notebook will use different config files 
through which we can incorporate different augmentation effects on our train
dataset.


## Experiments
The experiments folder will be organized as follows:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - EXP-01/ - create a new folder for each experiment you run
    - EXP-02/ - create a new folder for each experiment you run
    - EXP-03/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

The instructions for carrying out an experiment are given here:

### Edit the config file

Now we are ready for training. The Tf Object Detection API relies on 
**config files**. The config that we will use for this project is 
`pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. 
You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `./experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and 
validation files, as well as the location of the label_map file, pretrained 
weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file, `pipeline_new.config`, will be created in the root folder.

### Training

We can now launch an experiment with the Tensorflow object detection 
API. Move the `pipeline_new.config` to the `./experiments/new_experiment` folder. 
Configure it with appropriate augmentations and hyperparameters for training.
Now from the root directory of the project launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/new_experiment/ --pipeline_config_path=experiments/new_experiment/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/new_experiment/ --pipeline_config_path=experiments/new_experiment/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be 
ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running 
`python -m tensorboard.main --logdir experiments/new_experiment/`.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/new_experiment/pipeline_new.config --trained_checkpoint_dir experiments/new_experiment/ --output_directory experiments/new_experiment/exported/
```

This should create a new folder `experiments/new_experiment/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. 
To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/new_experiment/exported/saved_model --tf_record_path /data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/new_experiment/pipeline_new.config --output_path animation.gif
```
