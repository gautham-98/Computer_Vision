import os
import numpy as np
import cv2
import logging
logger = logging.getLogger('Exercise 2 Logging')
logger.setLevel(logging.DEBUG)
logging.debug("Init")

from perspectiveCorrection import *

# CONSTANTS (you are allowed to change those)
IMG_WIDTH = 400
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_AQUAMARINE = (51, 103, 236)

CORRECTOR = PerspectiveCorrector()


################################################################
# ABSTRACT CLASSES FOR  KeypointDetector AND GIVEN IMPLEMENTATIONS
################################################################

class AbstractKeypointDetector(object):
    '''
    Abstract Keypoint class
    '''
    def detect_keypoints(self, image):
        '''
        Input:
            (image) -- gray image with values between [0, 255]
        Output:
            List of detected keypoints. Fill the cv2.KeyPoint struct object 
            with keypoint coordinate (pt), main (angle) of the gradient
            in degrees and the detector (response). Set the (size) to 10.
        '''
        raise NotImplementedError()

class SIFTKeypointDetector(AbstractKeypointDetector):
    '''
    SIFT keypoint detection. See lecture slides. Similar but not the same as Harris Corner Detector.
    You could use this for reference. Note that the SIFT detector accounts for scale.
    '''
    def detect_keypoints(self, image):
        '''
        Input:
            (image) -- gray image with values between [0, 255]
        Output:
            List of detected keypoints. Fill the cv2.KeyPoint struct object 
            with keypoint coordinate (pt), main (angle) of the gradient
            in degrees and the detector (response). Set the (size) to 10.
        '''
        sift = cv2.SIFT_create()

        # Note that sift.detect outputs keypoints in the KeyPoint struct already.
        # You have to do this manually in your Harris Corner implementation.
        kp = sift.detect(image,None)

        # you might wanna check what happens without orientation in the SIFT case
        # for f in kp:
        #    logging.info(f.angle)

        return kp

class StupidKeypointDetector(AbstractKeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use for reference, what happens if your keypoints are not correct.
    '''
    def detect_keypoints(self, image):
        '''
        Input:
            (image) -- gray image with values between [0, 255]
        Output:
            List of detected keypoints. Fill the cv2.KeyPoint struct object 
            with keypoint coordinate (pt), main (angle) of the gradient
            in degrees and the detector (response). Set the (size) to 10.
        '''
        keypoints = []
        height, width = image.shape[:2]

        # loop over the image dimensions
        for y in range(height):
            for x in range(width):
                if int(image[y, x] + x + y) % 100 == 1:
                    # Fill cv2 Keypoint struct - do that also for your other implementations
                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    # Note that angle is not set here. 
                    # To account for rotations you need to set this as well in your implementation
                    f.angle = 0
                    # Dummy response.
                    f.response = 10

                    keypoints.append(f)

        return keypoints
    

################################################################
# ABSTRACT CLASSES FOR  KeypointDescriptor AND GIVEN IMPLEMENTATIONS
################################################################
    
class AbstractKeypointDescriptor(object):
    # Implement in child classes
    def describe_features(self, image, keypoints):
        '''
        Input:
            image -- gray image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc --  #keypoint x feature-descriptor-dimension numpy array
        '''
        raise NotImplementedError
    

class SIFTKeypointDescriptor(AbstractKeypointDescriptor):
    # A good reference descriptor
    def describe_features(self, image, keypoints):
        '''
        Reference implementation SIFT keypoint descriptors. 
        Note that this descriptor accounts for scale and orientation (provided by keypoint)
        Input:
            image -- gray image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc --  #keypoint x 128 (SIFT) numpy array
        '''

        desc = np.zeros((len(keypoints), 128))
        sift = cv2.SIFT_create()
        kp,siftdesc = sift.compute(image, keypoints)
        desc = np.array(siftdesc)
        return desc
    

################################################################
# ABSTRACT CLASSES FOR  KeypointDescriptor
################################################################

class AbstractKeypointMatcher(object):
    def match_features(self, desc1, desc2, num_matches=-1):
        '''
        Input:
            desc1 -- feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            num_matches -- Number of matches; Hard requirement to return not more than this number;
                num_matches=-1: return all matches
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError



################################
# HELPER FUNCTIONS
################################

# helper function: load image from filename
def load_image(filename):
    image = None
    if filename and os.path.isfile(filename):
        image = cv2.imread(filename)

        #change to RGB
        image = image[...,::-1]

        # resize image to fix WIDTH for faster processing
        logger.info(f"Image shape: {image.shape}")
        maxLength = max(image.shape)
        scale_resize = IMG_WIDTH/maxLength
        height = int(image.shape[0] * scale_resize)
        width = int(image.shape[1] * scale_resize)
        dim = (width, height)
        logger.info(f"New image dim: {dim}")
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    else:
        logger.error("Path %s not found." % filename)
    return image

# helper function: load two image for stitching
def load_image_pairs(imgPair):
    #try:
    images_arr =[None]*2

    if imgPair == "dom":
        IMAGE_LEFT = "images/img_dom_1.jpg"
        IMAGE_RIGHT= "images/img_dom_2.jpg"
    elif imgPair == "dom_rot":
        IMAGE_LEFT = "images/img_dom_1.jpg"
        IMAGE_RIGHT= "images/img_dom_2_rot.jpg"
    elif imgPair == "rose":
        IMAGE_LEFT = "images/img_rose_1.jpg"
        IMAGE_RIGHT= "images/img_rose_2.jpg"
    elif imgPair == "castle":
        IMAGE_LEFT = "images/img_castle_1.jpg"
        IMAGE_RIGHT= "images/img_castle_2_rot15.jpg"
    elif imgPair == "checkboard":
        IMAGE_LEFT = "images/checkboard.jpg"
        IMAGE_RIGHT= "images/checkboard_rot.jpg"
    elif imgPair == "rome":
        IMAGE_LEFT = "images/rome_1.jpg"
        IMAGE_RIGHT= "images/rome_2.jpg"
    else:
        raise FileNotFoundError("No images with pair name %s" % imgPair)

    images_arr[0] = load_image(IMAGE_RIGHT)
    images_arr[1] = load_image(IMAGE_LEFT)
    return images_arr



# helper function: load two image for stitching
def compute_homography(keypoints_arr, matches):
    logger.info('Estimating Homography...')
    M, hmask = CORRECTOR.computeHomography(keypoints_arr[0], keypoints_arr[1], matches)
    logger.info(f'Estimated Homography: {M}')
    return M


# helper function: extract keypoints based on an implemented DETECTOR
def compute_keypoints(img, DETECTOR):
    logger.info('Computing keypoint descriptors ...')
    keypoints = None
    if img is not None:
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints = DETECTOR.detect_keypoints(grayImage)
    else:
        logger.error('No image loaded!')
    return keypoints

# helper function: add keypoint descriptions based on an implemented DESCRIPTOR
def describe_keypoints(images_arr, keypoints_arr, DESCRIPTOR):
    logger.info('Computing keypoint descriptors ...')
    descriptors_arr=[None]*2
    for i in range(2):
        image = images_arr[i]#.astype(np.float32)
        #image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        descriptors_arr[i] = DESCRIPTOR.describe_features(grayImage, keypoints_arr[i])
    return descriptors_arr

# helper function: compute matches between arrays of keypoint descriptors based on implemented DESCRIPTOR
def compute_matches(descriptors_arr, MATCHER, num_matches=-1): # num_matches=-1 use all matches
    logger.info('Computing matches ...')
    matches = MATCHER.match_features(descriptors_arr[0], descriptors_arr[1], num_matches)
    return matches


################################
# VISUALIZATION FUNCTIONS
################################

# vis function: concat two images for displaying
def concat_images(imgs):
    # Skip Nones
    imgs = [img for img in imgs if img is not None]
    maxh = max([img.shape[0] for img in imgs]) if imgs else 0
    sumw = sum([img.shape[1] for img in imgs]) if imgs else 0
    vis = np.zeros((maxh, sumw, 3), np.uint8)
    vis.fill(255)
    accumw = 0
    for img in imgs:
        h, w = img.shape[:2]
        vis[:h, accumw:accumw+w, :] = img
        accumw += w

    return vis


# vis function: draw keypoint matches between the images
def draw_matches(img1, kp1, img2, kp2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = concat_images([img1, img2])

    kp_pairs = [[kp1[m.queryIdx], kp2[m.trainIdx]] for m in matches]
    status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.circle(vis, (x1, y1), 5, COLOR_GREEN, 2)
            cv2.circle(vis, (x2, y2), 5, COLOR_GREEN, 2)
        else:
            r = 5
            thickness = 6
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), COLOR_RED, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), COLOR_RED, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), COLOR_RED, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), COLOR_RED, thickness)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), COLOR_AQUAMARINE)

    return vis

# vis function: visualize keypoint matches
def vis_matches(images_arr, keypoints_arr, matches):
    vis = draw_matches(images_arr[0], keypoints_arr[0], images_arr[1], keypoints_arr[1], matches)
    return vis

# vis function: draw keypoints
def vis_keypoints(images_arr, keypoints_arr):
    assert(len(images_arr)==2)
    assert(len(images_arr)==len(keypoints_arr))
    for idx in range(2):
        images_arr[idx] = cv2.drawKeypoints(images_arr[idx], keypoints_arr[idx], images_arr[idx], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis = concat_images([images_arr[0], images_arr[1]])
    return vis

# vis function: visualize homography, i.e. drawing image borders of image 1 in image 2
def vis_homography(images_arr, M):
    result = CORRECTOR.drawHomography(images_arr[0], images_arr[1], M)
    vis = concat_images([images_arr[0], result])
    return vis

# vis function: visualize the image stitch, i.e. displaying the two images 
# after the homography transformation.
def vis_stitch(images_arr, M):
    result = CORRECTOR.stitchImages(images_arr[0], images_arr[1], M)
    return result


""" 
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Show Keypoints":
            interpret_fields(values)
            image_arr = load_images(IMAGE_RIGHT, IMAGE_LEFT)
            keypoints_arr = []
            for img in image_arr:
                keypoints = compute_keypoints(img)
                keypoints_arr.append(keypoints)
            vis_keypoints(image_arr,keypoints_arr, window["image"])
        if event == "Show Matches":
            interpret_fields(values)
            image_arr = load_images(IMAGE_RIGHT, IMAGE_LEFT)
            keypoints_arr = [] 
            for img in image_arr:
                keypoints = compute_keypoints(img)
                keypoints_arr.append(keypoints)
            matches = compute_matches(image_arr, keypoints_arr)
            vis_matches(image_arr,keypoints_arr,matches, window["image"])
        if event == "Show Homography":
            interpret_fields(values)
            image_arr = load_images(IMAGE_RIGHT, IMAGE_LEFT)
            keypoints_arr = []
            for img in image_arr:
                keypoints = compute_keypoints(img)
                keypoints_arr.append(keypoints)
            matches = compute_matches(image_arr, keypoints_arr)
            M = compute_homography(keypoints_arr,matches)
            vis_homography(image_arr, M, window["image"])
        if event == "Show Stitch":
            interpret_fields(values)
            image_arr = load_images(IMAGE_RIGHT, IMAGE_LEFT)
            keypoints_arr = []
            for img in image_arr:
                keypoints = compute_keypoints(img)
                keypoints_arr.append(keypoints)
            matches = compute_matches(image_arr, keypoints_arr)
            M = compute_homography(keypoints_arr,matches)
            vis_stitch(image_arr, M, window["image"]) 
            
            """

