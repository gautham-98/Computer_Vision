## Computer Vision Exercise 2: Feature extraction and matching 

### What and Why

Detecting image features and matching pairs of features between images for image stitching.
You will learn:
* Feature detection using Harris Corner Detection
* Feature description (a simple 25x25 window vs. MOPS using orientation)
* Feature matching (SSD and ratio)
* Advantages and drawbacks of different combinations

More detailed description of your tasks can be found on [Moodle]().

### Steps

1.  Implement feature matching using SSD and the Ratio Distance.
2.  Implement the most important code parts for Harris corner detection.
3.  Implement a simple 25x25 feature descriptor.
4.  Implement the MOPS (Multi-Scale Oriented Patches) Descriptor which makes additional use of the keypoint orientation.
5.  Being able to answer some questions.
  
### Exercise Structure

| Name                      | Function                                              |
| ------------              | ------------------------------------------------------|
| /images                   | Various image pairs to test your implementation.      |
| /ex2media/media           | Images for the notebook markdown cells.               |
| ex2.ipynb                 | Python notebook describing and implementing the tasks.|
| helper.py                 | Various helper and visu functions / class defs.       |
| perspectiveCorrection.py  | Homography estimation and transformation.             |
| README.md                 | README, This file.                                    |

#### Libraries recommended (you can also choose to implement with other libraries)
* see requirements.txt in exercise root folder

