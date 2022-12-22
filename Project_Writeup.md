## Submission Template

### Project overview

This project involves using the Tensorflow Object Detection API to identify cars, pedestrians and cyclists. Object Detection is an integral part of a self-driving car system as a self-driving car needs to perceive its surroundings and interpret the objects around it accurately in order to make autonomous movements. The current project, which involves only 3 classes of objects, is obviously a highly simplified version of a realistic self-driving car's image identification unit. Despite its simplicity, it is however a toy model whose generalisation to many more classes of objects could become part of a realistic self-driving machine.

### Set up
Please refer to `readme.md` in the project root folder.

### Dataset

#### Dataset analysis

This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.

The dataset contains 1719 images in total across all tfrecords in the data/train folder. Some sample images with bounding boxes of objects are shown below. The color code is: red for vehicle,  blue for pedestrian and green for cyclist.
![Alt Text](writeup_images/Images_Example.png)

We observe the occurrence frequency to be the highest for vehicles (29710), followed by pedestrians (8355), followed by cyclists (214). As such there definitely is strong imbalance
in the classes that appear in our dataset.
<img src="writeup_images/class_frequency_hist.png" alt="drawing" width="300"/>


The basic statistics for each of these classes are shown below. To get an idea about the variability of aspect ratio (= width/height) and diagonal-size of bounding boxes for each class (in pixels), we plot the individual class histograms as well:

##### Vehicle
<img src="writeup_images/df_1.png" alt="drawing" width="600"/>
<img src="writeup_images/vehicle_AR.png" alt="drawing" width="600"/>
<img src="writeup_images/vehicle_SIZE.png" alt="drawing" width="600"/>


##### Pedestrian
<img src="writeup_images/df_2.png" alt="drawing" width="600"/>
<img src="writeup_images/pdestrian_AR.png" alt="drawing" width="600"/>
<img src="writeup_images/pedestrian_SIZE" alt="drawing" width="600"/>

##### Cyclist
<img src="writeup_images/df_4.png" alt="drawing" width="600"/>
<img src="writeup_images/cyclist_AR.png" alt="drawing" width="600"/>
<img src="writeup_images/cyclist_SIZE.png" alt="drawing" width="600"/>


### Training

#### Reference experiment

The reference training is directly run with config file produced after running `edit_config.py` file.

- Batch Size = 2
- num_steps = 2500
- checkpoint_every_n = 500

We have the following result from training:
![Alt Text](writeup_images/reference-loss.png)

The evaluation plots are shown below:
![Alt Text](writeup_images/reference-Recall.png) ![Alt Text](writeup_images/reference-Precision.png)

The eval result side by side on image 1 is:
![Alt Text](writeup_images/reference-Eval_1.png)

Evidently no bounding boxes are detected.

The inference video also shows no bounding boxes.
![Alt Text](writeup_images/animation_reference.gif)

#### Improve on the reference
We use data augmentation to improve on the reference experiment. The augmentations employed are
```
data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    	}
    }
  data_augmentation_options{
      random_rgb_to_gray{
      probability: 0.2
      }
    }
  data_augmentation_options{
      random_horizontal_flip{
      probability: 0.2
      }
  }
  data_augmentation_options{
  	random_adjust_brightness{
    max_delta:0.3
    }
  }
 data_augmentation_options{
  	random_adjust_contrast{
    min_delta:0.7
    max_delta: 1.4
    }
  }
  data_augmentation_options{
  	random_adjust_hue{
	max_delta: 0.05
    }
  }
  data_augmentation_options{
  	random_image_scale{
    min_scale_ratio: 0.8
    max_scale_ratio: 1.3
    }
  }
```
The above augmentations were employed in the following experiments.

From the `Explore augmentations.ipynb` notebook, we can check out some sample images after these augmentation effects.

<img src="writeup_images/sample_aug1.png" alt="drawing" width="300"/>
<img src="writeup_images/sample_aug2.png" alt="drawing" width="300" height="580"/>

##### EXP-01

**Changes to the config file**

- `batch_size: 4`
- `num_steps: 2500`

**Runtime Modification**
- `checkpoint_every_n = 1000`

We have the following result from training:
![Alt Text](writeup_images/EXP-01-loss.png)

The evaluation plots are shown below:
![Alt Text](writeup_images/EXP-01-Recall.png) ![Alt Text](writeup_images/reference-Precision.png)

The eval result side by side on image 1 is:
![Alt Text](writeup_images/EXP-01-Eval_1.png)

The inference video starts to show some bounding boxes.
![Alt Text](writeup_images/animation_exp_01_night.gif)

##### EXP-02

**Changes to the config file**

- `batch_size: 8`
- `num_steps: 3000`
-  Added `aspect_ratios: 0.33`
-  Changed `scales_per_octave: 3`

**Runtime Modification**
- `checkpoint_every_n = 1000`

We have the following result from training:
![Alt Text](writeup_images/EXP-02-loss.png)

The evaluation plots are shown below:
![Alt Text](writeup_images/EXP-02-Recall.png) ![Alt Text](writeup_images/reference-Precision.png)

The eval result side by side on image 1 is:
![Alt Text](writeup_images/EXP-02-Eval_1.png)

The inference video also shows no bounding boxes.
![Alt Text](writeup_images/animation_exp_02_night.gif)
![Alt Text](writeup_images/animation_exp_02_day.gif)
![Alt Text](writeup_images/animation_exp_02_third.gif)
