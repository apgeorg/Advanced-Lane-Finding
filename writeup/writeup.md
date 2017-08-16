## Advanced Lane Finding 

In this project, the goal is to identify the lane boundaries in a video from a front-facing camera on a car. 
The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("bird's-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/chessboard_undist.png "Undistorted"
[image2]: ./images/undist.png "Undistorted"
[image3]: ./images/warp.png "Warped Image"
[image4]: ./images/binary.png "Combined Binary Image"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ../advanced_lane_lines_output.mp4 "Video"

### Camera Calibration

#### 1. Camera matrix and distortion coefficients

I start preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Then I compute the camera calibration and distortion coefficients. Here is an example of a distortion corrected calibration image. 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion correction 

As a first step I apply the distortion correction while using the calculated camera matrix and distortion coefficients from above. The following figure shows an distortion corrected image in which we can see a difference of the hood of the car at the bottom of the image.

![alt text][image2]

#### 2. Perspective transformation

As Next, I apply perspective transform which maps the points in a given image to different, desired, image points with a new perspective. In our case we are interested to get a bird’s-eye view that let us view a lane from above. For that, I hardcode the source and destination points as following:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 450      | 300, 0        | 
| 190, 720      | 300, 720      |
| 1130, 720     | 980, 720      |
| 690, 450      | 980, 0        |

Here's an example of my output for this step.

![alt text][image3]

#### 3. Color thresholds 

I used color thresholds on different color spaces to generate a binary image. The color mask is splitted in a yellow and a white mask. I've converted the image to RGB and HLS space. The thresholds are chosen on emperical results. For the yellow mask I used the thresholds below:

|  Color Space  | Lower thres.  | Upper thres.  |  
|:-------------:|:-------------:|:-------------:| 
| RGB           | (225,180,0)   | (255,255,170) | 
| HLS           | (20,120,80)   | (45,200,255)  |

For the white mask the follwing thresholds are applied:

|  Color Space  | Lower thres.  | Upper thres.  |  
|:-------------:|:-------------:|:-------------:| 
| RGB           | (100,100,200) | (255,255,255) | 

Both are applied to the warped image. So the resulting binary image looks like this:

![alt text][image4]

#### 4. Finding lane lines

I first calculate a histogram along all the columns in the lower half of the image. 

![alt text][image5]

Then I use a sliding window algorithm as mentioned in the lessons and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Curvature

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### Test on project video

The pipeline was applied on the provided project video and the final video result was quite well without any catastrophic failures that would cause the car to drive off the road.

Here's a [link to my video result](../advanced_lane_lines_output.mp4)

---

### Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

This alogrithm has problems identifying the lane lines in the challenge videos, because of the contrast and the fact that the lane lines are not as clearly visible as in the project video. Possible improvements:



