import cv2
import numpy as np


## Apply Homography ############################################################

# Minimum match count needed for an homography.
MIN_MATCH_COUNT = 8

class PerspectiveCorrector(object):
    def computeHomography(self, kp1, kp2, matches):
        '''
        Input:
            kp1, kp2 -- Keypoint lists of both images
            matches - Matches computed.
        Output:
            M - Estimated homography matrix M based on the keypoint matches. Uses RANSAC.
        '''
        M = None
        matchesMask = None
        if len(matches)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
        else:
            raise Exception( "Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT) )
        return M, matchesMask

    def drawHomography(self, img1, img2, M):
        '''
        Draws the transformed image border in the second image.
        Input:
            img1, img2 -- The two images
            M - Homography between the two images
        Output:
            img2 - Image 2 with transformed border from image 1.
        '''
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #print(pts)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        return img2

    def stitchImages(self, img1, img2, M):
        '''
        Stitches two images together using the homography M.
        Input:
            img1, img2 -- The two images
            M - Homography between the two images
        Output:
            result - concatenated image where img1 and img2 
            are stitched together based on the homography M
        '''
        # otherwise, apply a perspective warp to stitch the images
		# together
        width = img1.shape[1] + img2.shape[1]
        height = img1.shape[0] + 100
        result = cv2.warpPerspective(img1, M, (width, height))
        result[0:img2.shape[0], 0:img2.shape[1]] = img2
		# check to see if the keypoint matches should be visualized
        return result
        