import glob
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import utils.color as color 

"""
    Returns camera matrix and distortion coefficients.  
"""
def calibrate_camera(glob_str, nx=9, ny=6):
    # List of calibration images
    filenames = glob.glob(glob_str)
    # Prepare object points
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane. 
    for fname in filenames:
        # Read image and convert to grayscale
        img = cv2.imread(fname)
        gray = color.cvtColor(img, color.ConvertColor.BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, draw corners
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist
     
"""
    Performs image distortion correction and returns the undistorted image.
"""
def undistort(image, mtx, dist):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist
    
"""
    Returns a warped image, the perspective transform matrix and the inverse. 
"""
def warp(image):
    # Grab the image shape
    img_height, img_width = (image.shape[0], image.shape[1])
    # Define source/ destination points
    offset = 300
    #src = np.float32([(290, 660), (1020, 660), (595, 450), (690, 450)])
    src = np.float32([(190, 720), (1130, 720), (595, 450), (690, 450)])
    dst = np.float32([(offset, img_height), 
                      (img_width-offset, img_height), 
                      (offset, 0), 
                      (img_width-offset, 0)])
                      
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Calculate the inverse perspective transform matrix
    M_inv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image 
    warped = cv2.warpPerspective(image, M, (img_width, img_height))
    return warped, M, M_inv 
    
"""
    Returns an unwarped image with drawn lanes.
"""
def weighted_img(initial_image, image, M_inv, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (M_inv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (initial_image.shape[1], initial_image.shape[0])) 
    # Combine the result with the original image
    return cv2.addWeighted(initial_image, 1, newwarp, 0.3, 0)

"""
    Adds text to an image. 
"""    
def add_text(original_img, curv, center):
    img = np.copy(original_img)
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Radius of curvature: ' + '{:04.2f}'.format(curv) + 'm'
    cv2.putText(img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    text = 'Vehicle offset from lane center: ' + '{:04.1f}'.format(center) + 'm '  
    cv2.putText(img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return img

"""
    Converts to binary image
"""
def convert_2_binary(image, thresh):
        binary_output = np.zeros_like(image)
        binary_output[(image >= thresh[0]) & (image <= thresh[1])] = 1
        return binary_output

"""
    Applies color mask to an image
"""
def apply_color_mask(image, lower_thres, upper_thres):
    mask = cv2.inRange(image, lower_thres, upper_thres)
    img = cv2.bitwise_and(image, image, mask=mask).astype(np.uint8)
    return img, mask
    
"""
    Applies color thresholds and returns the mask and resulted image.
"""    
def color_thresh(image):
    # rgb for yellow lines
    rgb_yellow, mask = apply_color_mask(image, np.array([225,180,0], dtype="uint8"), np.array([255, 255, 170], dtype="uint8"))
    rgb_yellow = convert_2_binary(cv2.cvtColor(rgb_yellow, cv2.COLOR_RGB2GRAY), [20, 255])
    
    # rgb for white lines
    rgb_white, mask = apply_color_mask(image, np.array([100,100,200], dtype="uint8"), np.array([255, 255, 255], dtype="uint8"))
    rgb_white = convert_2_binary(cv2.cvtColor(rgb_white, cv2.COLOR_RGB2GRAY), [20, 255])
        
    # hls for yellow lines
    hls_yellow, mask = apply_color_mask(cv2.cvtColor(image, cv2.COLOR_RGB2HLS), np.array([20,120,80], dtype="uint8"), np.array([45, 200, 255], dtype="uint8"))
    hls_yellow = cv2.cvtColor(hls_yellow, cv2.COLOR_HLS2RGB)
    hls_yellow = convert_2_binary(cv2.cvtColor(hls_yellow, cv2.COLOR_RGB2GRAY), [20, 255])

    binary_image = np.zeros_like(hls_yellow)
    binary_image[(hls_yellow == 1) | (rgb_yellow == 1) | (rgb_white == 1)] = 1
    return binary_image
    
    
def fit_lines(image, draw_lines=False):
    histogram = np.sum(image[image.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((image, image, image))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(image.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window+1)*window_height
        win_y_high = image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if draw_lines == True:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(image, cmap="gray")
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Calculate car center
    car_center = image.shape[1]/2
    h = image.shape[0]
    left_fit_x = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
    right_fit_x = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
    lane_center = (right_fit_x + left_fit_x) /2
    center = (car_center - lane_center) * xm_per_pix
    # Now our radius of curvature is in meters
    #print(left_fit_x, 'm', right_fit_x, 'm')
    return ploty, left_fitx, right_fitx, left_curverad, right_curverad, center
