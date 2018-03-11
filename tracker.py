import numpy as np
import cv2

# ### Detects lane pixels and fit to a polynomial function to find the lane boundary.

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # y values of the last n fits of the line
        self.recent_yfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients of the fitted line of averaged x values of the fitted lines of last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = [] 
        #y values for detected line pixels
        self.ally = []
        
        self.yvals = None
        
    def fit_line(self, smooth_factor, margin):
        allx_smooth = np.concatenate(self.allx[-smooth_factor:]).ravel()
        ally_smooth = np.concatenate(self.ally[-smooth_factor:]).ravel()
        # Fit a second order polynomial to pixel positions in each lane line
        if(len(self.allx[-1])>350):
            previous_fit = self.current_fit
            self.current_fit = np.polyfit(ally_smooth, allx_smooth, 2)
            if self.detected:
                self.diffs = self.current_fit - previous_fit
            fitx = self.current_fit[0]*self.yvals*self.yvals + self.current_fit[1]*self.yvals + self.current_fit[2]
            fitx = np.array(fitx,np.int32)

            if fitx is not None:
                self.detected = True
                self.recent_xfitted.append(fitx)
                self.bestx = np.average(self.recent_xfitted[-smooth_factor:],axis = 0)
                self.best_fit = np.polyfit(self.yvals, self.bestx, 2)
        

class LineTracker():

    # when starting a new instance please be sure to specify all unassigned variables
    def __init__(self, window_width, window_height, margin, ym = 1, xm =1, smooth_factor=15, lane_width=3.7):


        # the window pixel width of the center values, used to count pixels inside center windows to determine curve values
        self.window_width = window_width

        # the window pixel height of the center values, used to count pixels inside center windows to determine curve values
        # breaks the image into vertical levels
        self.window_height = window_height

        # The pixel distance in both directions to slide (left_window + right_window) template for searching
        self.margin = margin

        self.ym_per_pix = ym # meters per pixel in vertical axis

        self.xm_per_pix = xm # meters per pixel in horizontal axis
        
        self.smooth_factor = smooth_factor
        
        self.lane_width = lane_width
        
        self.left_line = Line()
        self.right_line = Line()

    # the main tracking function for finding and storing lane segment positions
    def find_lane_pixels(self, warped):

        # window settings
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin # How much to slide left and right for searching

        window = np.ones(window_width) # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum one third from bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(2*warped.shape[0]/3):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(2*warped.shape[0]/3):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Go through each layer looking for max pixel locations
        for level in range((int)(warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            if np.max(conv_signal[l_min_index:l_max_index]) > 10000:
                l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))  
            r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
            if np.max(conv_signal[r_min_index:r_max_index]) > 10000:
                r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Identify window boundaries in x and y (and right and left)
            win_y_low = int(warped.shape[0]-(level+1)*window_height)
            win_y_high = int(warped.shape[0]-(level)*window_height)
            win_xleft_low = l_center - offset
            win_xleft_high = l_center + offset
            win_xright_low = r_center - offset
            win_xright_high = r_center + offset
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        self.left_line.allx.append(nonzerox[left_lane_inds])
        self.left_line.ally.append(nonzeroy[left_lane_inds])
        self.right_line.allx.append(nonzerox[right_lane_inds])
        self.right_line.ally.append(nonzeroy[right_lane_inds])
            
        return self.left_line, self.right_line
    
    
    def find_lines(self,warped):
        self.find_lane_pixels(warped)
        yvals = range(0,warped.shape[0])
        
        self.left_line.yvals = yvals
        self.right_line.yvals = yvals
    
        # Fit a second order polynomial to pixel positions in each lane line
        self.left_line.fit_line(self.smooth_factor, self.margin)            
        self.right_line.fit_line(self.smooth_factor, self.margin)

        if self.left_line.detected:
            self.left_line.line_base_pos = (self.left_line.bestx[-1]-warped.shape[1]/2)*self.xm_per_pix
        if self.right_line.detected:
            self.right_line.line_base_pos = (self.right_line.bestx[-1]-warped.shape[1]/2)*self.xm_per_pix
        
        return self.left_line, self.right_line
    
    # Note: This returns a bird eye view of road.
    def get_road_img(self, warped):
        # Create an image to draw on and an image to show the selection window
        road_img = np.zeros_like(warped)
        road_img = np.array(cv2.merge((road_img,road_img,road_img)),np.uint8)
            
        if self.left_line.detected and self.right_line.detected:
            # Fit a second order polynomial to each
            left_fit = self.left_line.best_fit
            right_fit = self.right_line.best_fit
            # Generate x and y values for plotting
            ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

   
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))
            lane_pts = np.hstack((left_line_window2, right_line_window1))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(road_img, np.int_([left_line_pts]), (255,0, 0))
            cv2.fillPoly(road_img, np.int_([right_line_pts]), (0,0, 255))
            cv2.fillPoly(road_img, np.int_([lane_pts]), (0,255, 0))
        
        return road_img

