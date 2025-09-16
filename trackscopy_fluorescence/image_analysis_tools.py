# -*- coding: utf-8 -*-
"""

- This is a module to process color channels of fluorescence images 
  separately and generate the corresponding trajectories. 

- Image processing functions include Background correction, Segmentation,
  Connected-Regions Estimation and Tracking

- The basic algorithm for single-channel image processing framework is based on:

    [1] J. Crocker, D. Grier, Methods of Digital Video Microscopy for Colloidal Studies, Journal of Colloid and
        Interface Science, 179, 1 (1996)

    [2] M. Theves, J. Taktikos, V. Zaburdaev, H. Stark, C. Beta, A Bacterial Swimmer with Two Alternating
        Speeds of Propagation, Biophysical Journal, 105,8 (2013)

    with some additional features like normalization of brightness and contrast,
    fixing over-exposed pixels and blurring for signal enhancement. 

- Author: Agniva Datta @ University of Potsdam, Germany

"""

# =======================
# Import Required Modules
# =======================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cv2 as cv
import tifffile as tf
from numba import jit
from numba.typed import List
import skimage.morphology as mo
import skimage.measure as ms
from tqdm.auto import tqdm
import concurrent.futures
import pickle
import os
import shutil
import pandas as pd
from datetime import datetime
from scipy.signal import savgol_filter

# ======================================================
# Load default parameters from external parameter script
# ======================================================

with open('trackscopy_fluorescence/default_parameters_image_analysis.py') as f:
    exec(f.read())

# ======================
# Image Analysis Class
# ======================

class Image_Analysis:
    """
    A class for analyzing microscopy fluorescence images:
    Includes preprocessing, segmentation, region detection, and tracking.
    """

    def __init__(self, image, parameters=default_parameters_image_analysis):
        """
        Constructor: initializes with input image stack and parameters.

        Parameters:
        - image: 3D array (frames x height x width) of raw microscopy images
        - parameters: tuple of parameters for analysis pipeline

        Initializes internal attributes for pixel fixing, normalization, stack dimensions, 
        resolution, max displacement for tracking, and bit-depth-based intensity scaling.
        """
        self.raw_image = fix_pixel_bug(image)
        self.raw_image = normalize_brightness_contrast(self.raw_image)

        self.nImg = len(self.raw_image)
        self.ht = len(self.raw_image[0])
        self.wt = len(self.raw_image[0][0])

        (self.framerate, self.magnification, self.pixel_size_camera, 
         self.max_particle_speed, self.memory_b, self.dim, 
         self.threshold_factor, self.blur_factor, 
         self.SMOOTH_TRACKS_LIM, self.SMOOTH_TRACKS_SPREAD, 
         self.nbins, self.R) = parameters

        self.resolution = self.magnification/self.pixel_size_camera
        self.maxdisp = np.floor(self.max_particle_speed*self.resolution/self.framerate)
        self.goodenough = 5

        original_dtype = self.raw_image.dtype
        bit_depth = np.iinfo(original_dtype).bits
        self.maxi = float(2**bit_depth - 1)
        
    def Analysis(self):
        """
        Run the full analysis pipeline: 
        Background correction → Segmentation → Connected regions → Tracking.

        Stores raw and smoothed tracks, and saves parameters for reproducibility.
        """
        self.background_correction()
        del self.raw_image
        self.segmentation()
        self.connected_regions_estimation()
        
        try:
            self.tracking()
            self.smooth_tracks = smooth_tracks(self.tracklist,self.SMOOTH_TRACKS_SPREAD,self.SMOOTH_TRACKS_LIM)
        except:
            print('Check segmentation: might be either too many/too less number of motile cells')
            
        self.save_parameters()
        
    def background_correction(self):
        """
        Correct uneven illumination/background across frames.

        Computes stack statistics, normalizes background, applies blurring,
        and stores corrected stack and minimum intensity projection.
        """
        imtype = self.raw_image.dtype
        (counts, intensities, Bg, Min, Max, stackHist) = calculate_stack_params(self.raw_image,self.maxi,self.nbins)
        
        Mincorr = correct_background(Min,Bg,self.maxi,[])
        Maxcorr = correct_background(Max,Bg,self.maxi,[])
        conversionRange = [Mincorr.min(), Maxcorr.max()]
        
        corr_images = np.zeros(self.raw_image.shape,imtype)
        
        for i in tqdm(range(self.nImg),desc='Background Correction'):
            Img = correct_background(self.raw_image[i],Bg,self.maxi,conversionRange)
            corr_images[i] = (Img*self.maxi).astype(imtype)
            corr_images[i] = cv.blur(corr_images[i],(self.blur_factor,self.blur_factor))
            
        self.background_corrected_image = Image(corr_images)
        (corrCounts, corrIntensities, CorrAvg, CorrStackMin, CorrStackMax, corrStackHist) = calculate_stack_params(corr_images,self.maxi,self.nbins)
        self.minimum_projection = CorrStackMin
        self.level = self.threshold_factor*threshold(corrStackHist,corrIntensities)

    def segmentation(self):
        """
        Segment corrected images using thresholding + morphology.

        Uses parallel processing to generate segmented stack and stores as Image object.
        """
        seg_images = np.zeros(self.background_corrected_image.im_array.shape,np.uint8)
        seg_images_list = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for Im in tqdm(executor.map(segmentation,self.background_corrected_image.im_array,[self.R]*self.nImg,[self.level]*self.nImg,[self.maxi]*self.nImg),desc='Segmentation'):
                seg_images_list.append(Im)
        
        for i in range(self.nImg):
            seg_images[i][:,:] = seg_images_list[i][:,:]
            
        self.segmented_image = Image(seg_images)
        
    def connected_regions_estimation(self):
        """
        Label and extract connected regions (cell candidates) in segmented images.

        Stores a list of connected region positions for tracking.
        """
        positionlist = connect(self.segmented_image.im_array)
        self.connected_position_list = positionlist
        
    def tracking(self):
        """
        Link detected regions across frames into tracks.

        Generates tracklist as [x, y, frame] arrays for each tracked particle.
        """
        self.tracklist = []
        tracks = track(self.connected_position_list,self.dim,self.maxdisp,self.goodenough,self.memory_b)
        num_tracks = int(max(tracks[:,-1]))
        
        for i in tqdm(range(num_tracks),desc='Saving tracks'):
            find = np.where(tracks[:,-1]==i+1)[0]
            num_cor = len(find)
            Traj = np.zeros([num_cor,3],float)
            Traj[:,0:2] = tracks[find][:,0:2]
            Traj[:,2] = tracks[find][:,-2]
            self.tracklist.append(Traj)
            
    def save_parameters(self):
        """
        Save analysis parameters in dictionary form.

        Includes framerate, magnification, resolution, tracking limits, and thresholds.
        """
        self.parameter_dict = {
            "framerate": self.framerate,
            "magnification": self.magnification,
            "pixel_size_camera":  self.pixel_size_camera,
            "maximum_particle_speed":  self.max_particle_speed,
            "memory_b": self.memory_b,
            "dim": self.dim,
            "threshold_factor": self.threshold_factor,
            "blur_factor": self.blur_factor,
            "SMOOTH_TRACKS_LIM": self.SMOOTH_TRACKS_LIM,
            "SMOOTH_TRACKS_SPREAD": self.SMOOTH_TRACKS_SPREAD,
            "nbins": self.nbins,
            "resolution": self.resolution,
            "R": self.R,
            "maxdisp": self.maxdisp,
            "goodenough": self.goodenough,
        }
        
    def save(self, path):
        """
        Save the full analysis object as pickle file.

        Parameters:
        - path: file path to save the object
        """
        dbfile = open(path, 'wb')
        pickle.dump(self, dbfile)
        dbfile.close()
        
    def save_tracks_only(self,path):
        """
        Save only tracklist and parameters (without raw images).

        Parameters:
        - path: file path to save tracks
        """
        save_tracklist(self.tracklist,self.parameter_dict,path)


# ===================
# Image Helper Class
# ===================

class Image:
    """
    Object for image stack arrays.
    """
    def __init__(self,image):
        """
        Initialize Image object.

        Parameters:
        - image: 3D array (frames x height x width)
        """
        self.im_array = image
        self.nImg = len(image)
        self.ht = len(image[0])
        self.wt = len(image[0][0])

    def show_frame(self,i):
        """
        Show a given frame with matplotlib.

        Parameters:
        - i: frame index to display
        """
        plt.imshow(self.im_array[i],cmap='gray')
        
    def save_image(self,path):
        """
        Save image stack as TIFF file.

        Parameters:
        - path: file path to save the stack
        """
        tf.imwrite(path,self.im_array)
        

# =========================
# Save Tracks Helper Class
# =========================

class save_tracklist:
    """
    Save tracklist and associated parameters as pickle file.
    """
    def __init__(self,tracklist,params,path):
        """
        Initialize and save tracklist.

        Parameters:
        - tracklist: list of track arrays [x, y, frame]
        - params: analysis parameters dictionary
        - path: file path to save pickle
        """
        self.tracks = tracklist
        self.path = path
        self.parameter_dict = params
        self.save()
        
    def save(self):
        """
        Save tracklist object as pickle file.
        """
        dbfile = open(self.path, 'wb')
        pickle.dump(self, dbfile)
        dbfile.close()

# ===================
# Utility Functions
# ===================

def fix_pixel_bug(sample_image):
    """
    Fix potential pixel artifacts.

    If the second-highest histogram bin is empty, replace extreme pixel values with mean intensity.

    Parameters:
    - sample_image: 2D or 3D numpy array

    Returns:
    - corrected image array
    """
    numbers, bins = np.histogram(sample_image.flatten(),bins=10)
    if numbers[-2] == 0:
        sample_image[sample_image>bins[-2]]  =  np.mean(sample_image)
    return sample_image


def normalize_brightness_contrast(stack, apply_clahe=False, clahe_clip=2.0, clahe_grid=(8, 8)):
    """
    Normalize brightness across frames and optionally apply CLAHE.
    Automatically handles 8-bit or 16-bit input.

    Parameters:
    - stack: 3D numpy array of shape (frames, height, width), dtype=uint8 or uint16
    - apply_clahe: bool, whether to apply CLAHE for local contrast enhancement
    - clahe_clip: float, CLAHE clip limit (controls contrast amplification)
    - clahe_grid: tuple, CLAHE tile grid size

    Returns:
    - corrected_stack: same shape and dtype as input, brightness-normalized (and optionally CLAHE-enhanced)
    """
    
    # Get input image data type and its bit depth (e.g., 8-bit, 16-bit)
    original_dtype = stack.dtype
    bit_depth = np.iinfo(original_dtype).bits
    max_val = float(2**bit_depth - 1)  # maximum possible intensity value

    # Convert image stack to float for arithmetic operations
    stack_float = stack.astype(np.float32)
    num_frames = stack.shape[0]

    # Compute mean brightness per frame
    frame_means = stack_float.mean(axis=(1, 2))
    target_mean = np.median(frame_means)  # reference brightness (median of means)

    # Initialize normalized stack
    normalized = np.empty_like(stack_float)
    for i in range(num_frames):
        # Compute correction factor for each frame to match target_mean
        factor = target_mean / (frame_means[i] + 1e-8)
        # Scale and clip intensities
        normalized[i] = np.clip(stack_float[i] * factor, 0, max_val)

    if apply_clahe:
        # Convert normalized image to 8-bit for CLAHE (required by OpenCV)
        normalized_8bit = ((normalized / max_val) * 255).clip(0, 255).astype(np.uint8)

        # Create CLAHE object
        clahe = cv.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        enhanced_8bit = np.empty_like(normalized_8bit)
        # Apply CLAHE per frame
        for i in range(num_frames):
            enhanced_8bit[i] = clahe.apply(normalized_8bit[i])

        # Convert back to original depth (scale back from 8-bit)
        corrected = ((enhanced_8bit.astype(np.float32) / 255) * max_val).clip(0, max_val)
    else:
        corrected = normalized

    # Return corrected stack in original dtype
    return corrected.astype(original_dtype)


@jit(nopython=True)
def calculate_stack_params(images,maxi,n):
    """
    Compute global statistics of an image stack.

    Parameters:
    - images: list/array of frames
    - maxi: maximum possible intensity (depends on bit depth)
    - n: number of histogram bins

    Returns:
    - count: cumulative histogram counts across stack
    - intensity: bin edges of histogram
    - A: average image (mean across frames)
    - MIN: minimum projection image (pixelwise min across frames)
    - MAX: maximum projection image (pixelwise max across frames)
    - stack_Hist: histogram for each frame separately
    """
    
    Image = images[0]
    imtype = Image.dtype

    # Histogram of first image
    [count,intensity] = np.histogram(Image.ravel(),n,[0,maxi+1])
    nImage = len(images)

    # Initialize accumulators
    A = Image.astype(np.uint64)  # running sum of images
    MIN = Image                  # running min
    MAX = Image                  # running max

    # Store histogram per frame
    stack_Hist = np.zeros((len(count),nImage))
    stack_Hist[:,0] = count

    # Loop through all frames to accumulate stats
    for i in range(1,nImage):
        Image = images[i]
        A += Image.astype(np.uint64)
        MIN = np.minimum(MIN,Image)
        MAX = np.maximum(MAX,Image)

        # Update histogram counts
        newCount = np.histogram(Image.ravel(),n,[0,maxi+1])[0]
        stack_Hist[:,i] = newCount
        count += newCount

    # Compute mean image
    A = A/nImage
    A = A.astype(imtype)

    return (count, intensity, A, MIN, MAX, stack_Hist)


@jit(nopython=True)
def mat2gray(M, Range):
    """
    Normalize matrix M to [0,1] range given a specified intensity Range.

    Parameters:
    - M: input 2D array (image)
    - Range: [min, max] values for scaling

    Returns:
    - M: rescaled image (values outside range set to 0)
    """
    Mmin = Range[0]
    Mmax = Range[1]

    for i in range(len(M)):
        for j in range(len(M[0])):
            m = M[i,j]

            if m < Mmin:
                M[i,j] = 0
            if m >= Mmin and m < Mmax:
                M[i,j] =  (1.0/(Mmax-Mmin))*(M[i,j]-Mmin)
            if m >= Mmax:
                M[i,j] = 0

    return M


def correct_background(Image,Bg,maxi,conversionRange):
    """
    Correct background illumination using background image Bg.

    Parameters:
    - Image: input image (single frame)
    - Bg: estimated background image
    - maxi: maximum possible intensity value
    - conversionRange: optional scaling range for normalization

    Returns:
    - corrected image (float64)
    """
    # Normalize image and background to [0,1]
    Image = np.double(Image/maxi)
    Bg =  np.double(Bg/maxi)

    # Divide image by background to correct uneven illumination
    T = np.divide(Image,Bg)

    if len(conversionRange) == 0:
        return T
    else:
        # Convert conversionRange into numba List for mat2gray
        R = List()
        [R.append(x) for x in conversionRange]
        norm_image = mat2gray(T,R)
        return norm_image


@jit(nopython=True)
def max_entropy(h,x):
    """
    Compute threshold using maximum entropy method.

    Parameters:
    - h: histogram counts
    - x: number of intensity bins

    Returns:
    - threshold (normalized by number of bins)
    """
    # Normalize histogram into probability distribution
    p = h/sum(h)
    P = np.cumsum(p)  # cumulative probability

    H_max = 0
    threshold = 0

    # Loop over all candidate thresholds
    for t in range(len(p)):
        H_back = 0  # entropy of background
        for i in range(t):
            if p[i] != 0:
                H_back -= p[i]/P[t] *np.log(p[i]/P[t])

        H_obj = 0   # entropy of object
        for i in range(t,len(p)):
            if p[i] != 0:
                if P[t] != 1:
                    H_obj -= p[i]/(1-P[t]) *np.log(p[i]/(1-P[t]))

        # Total entropy = background + object
        H_tot = H_back + H_obj

        # Keep threshold with maximum entropy
        if H_max<H_tot:
            H_max = H_tot
            threshold = t

    return np.divide(threshold,x)


@jit(nopython=True)	
def threshold(stack,intensity):
    """
    Compute global threshold value for stack using max-entropy method.

    Parameters:
    - stack: histogram counts per frame (2D array)
    - intensity: intensity bin edges

    Returns:
    - median threshold across frames
    """
    framelevel = np.zeros(len(stack[0]))

    # Compute threshold for each frame
    for i in range(len(stack[0])):
        framelevel[i] = max_entropy(stack[:,i],len(intensity))

    # Return median threshold (robust against outliers)
    return np.median(framelevel)


'''def fillhole(im_in):
    # Alternative function (currently unused) to fill holes in binary mask
    im_floodfill = im_in.copy()
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    im_out = im_in | im_floodfill_inv
    return im_out'''


def segmentation(Img,R, level, maxi):
    """
    Segment an image using global threshold and morphological processing.

    Parameters:
    - Img: input image (2D)
    - R: radius of structuring element for morphology
    - level: threshold factor (multiplied by maxi)
    - maxi: maximum possible intensity

    Returns:
    - out: binary segmented image
    """
    # Structuring element for morphology
    se = mo.disk(R)

    # Kernel for erosion
    kernel = np.ones((5,5),np.uint8)

    # Apply global threshold
    thresh, out = cv.threshold(Img, level*maxi, maxi, cv.THRESH_BINARY)

    # Morphological opening (remove small noise)
    out = mo.opening(out,se)

    # Morphological closing (fill small holes)
    out = mo.closing(out,se)

    # Erode to refine boundaries
    out = cv.erode(out,kernel,iterations = 1)

    # Optionally: fill holes (commented out)
    # out = fillhole(out.astype('uint8'))
    
    return out


'''def segmentation(Img,R,level,maxi):
    # Alternative segmentation (inverted threshold + morphology)
    se = mo.disk(R)
    thresh, out = cv.threshold(Img, level*maxi, maxi, cv.THRESH_BINARY)
    out = (maxi-out)
    #out = mo.opening(out,se)
    out = mo.closing(out,se)
    kernel = np.ones((2,2),np.uint8)
    out = cv.erode(out,kernel,iterations=1)
    
    return out'''


def connect(seg_images):
    """
    Extract connected components (regions) from segmented image stack.

    Parameters:
    - seg_images: 3D numpy array (frames x height x width)

    Returns:
    - pos: array of detected region centroids, areas, and frame indices
           format = [x, y, area, frame]
    """
    nFrames = len(seg_images)
    poslist = [0.0,0.0,0.0,0.0]   # initialize placeholder
    
    proplist = {'centroid', 'area'}  # properties to extract
    
    for i in tqdm(range(nFrames),desc='Estimating connected regions:'):
        # Label connected regions
        Cc = ms.label(seg_images[i],connectivity=2)

        # Extract centroid and area per region
        stats = ms.regionprops_table(Cc,properties = proplist)
        
        if len(stats['area']) != 0:
            # Arrange centroid (x,y) and area
            templist = np.column_stack([stats['centroid-1'],stats['centroid-0'],stats['area']])
            
            # Sort by area
            templist = templist[stats['area'].argsort()]
            
            # Add frame index to each detection
            templist = np.column_stack([templist,(i+1)*np.ones(len(templist[:,0]))])
    
            # Append to global list
            poslist = np.row_stack([poslist,templist])
    
    # Remove placeholder row
    pos = np.delete(poslist,0,0)

    return pos


def track(xyzs,dim,maxdisp,goodenough,memory_b):

    """
    Custom single-particle tracking algorithm.

    Parameters:
    - xyzs: array of detected points per frame [x, y, area, frame]
    - dim: number of spatial dimensions (typically 2 for x,y)
    - maxdisp: maximum allowed displacement between frames
    - goodenough: minimum acceptable track length
    - memory_b: number of frames a track can be missing before termination

    Returns:
    - newtracks: array containing linked tracks with unique IDs
                 columns = [x, y, (optional extra features...), frame, track_id]
    """

    # --------------------------
    # Initialization
    # --------------------------

    dd = len(xyzs[0]) - 1     # last column index (frame index)
    t = xyzs[:,dd]            # extract frame numbers for each detection

    # Compute frame transitions
    st = np.roll(t,1)
    end = len(st)
    st = t[1:end] - st[1:end]
    w = np.where(st>0)[0]     # indices where time advances
    z = len(w)
    z += 1                    # total number of frame steps

    # Find indices where frame changes
    indices = np.where(t != np.roll(t,-1))[0]
    count = len(indices)
    res = indices
    res = np.insert(res,0,0)
    res = np.insert(res,len(res),len(t)-1)

    # Number of detections in first frame
    ngood = res[1] - res[0] + 1
    eyes = np.arange(0,ngood,1)      # indices of first-frame detections
    pos = xyzs[eyes,0:dim]           # positions of first detections

    # Initialize tracking parameters
    istart = 1
    n = ngood
    zspan = 50                       # number of frames per processing block
    if n > 200:
        zspan = 20
    if n > 500:
        zspan = 10

    # Matrices for storing assignments
    resx = np.zeros([zspan,n],int) - 1
    bigresx = np.zeros([z,n]) - 1
    mem = np.zeros([n,1])            # memory buffer per track

    # If minimum track length required, prepare arrays
    if goodenough > 0:
        dumphash = np.zeros([n,1])   # marks bad/short tracks
        nvalid = np.ones([n,1])      # counts how long track has been valid

    uniqid = np.arange(0,n,1)        # assign unique IDs to first detections
    maxid = n
    olist = [0.0,0.0]                # list of output track links
    resx[0,:] = eyes
    maxdisq = maxdisp**2             # squared displacement threshold

    Test = []  # debugging placeholder

    # --------------------------
    # Main loop over frames
    # --------------------------
    for i in range(istart,z):

        ispan = (i%zspan)
        m = res[i+1] - res[i]                # number of detections in this frame
        eyes = np.arange(0,m,1)
        eyes = eyes + res[i] + 1

        if m>0:

            xyi = xyzs[eyes,0:dim]           # detections in current frame
            found = np.zeros([m,1])          # whether detection has been linked

            wh = np.where(pos[:,0] >= 0)[0]  # active tracks from previous step
            ntrack = len(wh)

            # --------------------------
            # Construct cost matrices
            # --------------------------
            xmat = np.zeros([ntrack,m],int)
            ysize = max(m,ntrack)
            ymat = np.zeros([ysize,ysize],int)
            count = 0

            # Fill xmat indices
            for kk in range(ntrack):
                for ll in range(m):
                    xmat[kk,ll] = count
                    count += 1

            # Fill ymat indices
            for kk in range(m):
                for ll in range(ntrack):
                    ymat[kk,ll] = count
                    count += 1

            # Wrap indices for matrix dimensions
            xmat = xmat%m 
            if ntrack > 0:
                ymat = (ymat%ntrack ).transpose()
            [lenxn,lenxm] = list(xmat.shape)

            # --------------------------
            # Compute squared distances
            # --------------------------
            for d in range(dim):
                x = xyi[:,d]
                y = pos[wh,d]
                xm = x[xmat]
                ym = y[ymat[0:lenxn,0:lenxm]]

                if xm.shape != ym.shape:
                    xm = xm.transpose()

                if d == 0:
                    dq = (xm-ym)**2
                else:
                    dq = dq + (xm-ym)**2

            ltmax = dq < maxdisq   # boolean: within max displacement

            # --------------------------
            # Assignment by greedy matching
            # --------------------------
            rowtot = np.zeros([n,1])
            actual = ltmax.sum(axis=1)
            rowtot[wh] = np.array([actual],int).T

            if ntrack > 1:
                coltot = ltmax.sum(axis=0)
            else:
                coltot = ltmax

            coltot = coltot.ravel()
            which1 = np.zeros([n,1],int) - 1

            # Assign detections to closest existing track
            for j in range(ntrack):
                for k in range(len(ltmax[j,:])):
                    if ltmax[j,k] == max(ltmax[j,:]):
                        w = k
                        which1[wh[j]] = w

            w = np.where(rowtot == 1)[0]
            ngood = len(w)

            if ngood != 0:
                ww = np.where(coltot[which1[w]] == 1)[0]
                ngood = len(ww)
                if ngood != 0:
                    # Link detection to track
                    resx[ispan,w[ww]] = eyes[which1[w[ww]]].ravel()
                    found[which1[w[ww]]] = 1
                    rowtot[w[ww]] = 0
                    coltot[which1[w[ww]]] = 0

            # --------------------------
            # More complex matching (clusters of possible assignments)
            # --------------------------
            labely = np.where(rowtot>0)[0]
            ngood = len(labely)
            if ngood != 0:
                labelx = np.where(coltot>0)[0]
                nontrivial = 1
            else:
                nontrivial = 0

            if nontrivial == 1:
                # Compute bipartite graph of possible bonds between tracks and detections
                # (detailed Hungarian-like clustering logic follows)
                # Bonds store which detection could match which track
                xdim = len(labelx)
                ydim = len(labely)
                bonds = np.zeros([1,2])
                bondlen = [0.0]

                for j in range(ydim):
                    distq = np.zeros([xdim],float)
                    for d in range(dim):
                        distq = distq + ((xyi[labelx,d] - pos[labely[j],d])**2)

                    w = np.where(distq < maxdisq)[0]
                    w = w.transpose()
                    ngood = len(w)

                    newb = np.vstack((w,np.zeros([1,ngood])+j+1))
                    bonds = np.vstack((bonds,newb.transpose()))
                    distq  = distq[np.newaxis]
                    distq = distq.T
                    bondlen = np.vstack((bondlen,distq[w]))

                # Drop dummy first row
                bonds = bonds[1:len(bondlen),:]
                bondlen = bondlen[1:len(bondlen)]
                numbonds = len(bonds[:,0])
                mbonds = np.zeros(bonds.shape)
                mbonds[:,:] = bonds[:,:]
                max_dim = max([xdim,ydim])

                # If few bonds: trivial cluster
                if max_dim < 4:
                    nclust = 1
                    maxsz = 0
                    bmap = np.zeros([len(bonds[:,0]+1),1]) - 1

                else:
                    # Otherwise: cluster assignment search
                    # (lengthy logic: BFS-like expansion of candidate assignments)
                    lista = np.zeros(numbonds)
                    listb = np.zeros(numbonds)
                    nclust = 0
                    maxsz =  0
                    thru = xdim

                    while thru != 0:
                        w = np.where(bonds[:,1] >= 0)[0]
                        lista[0] = bonds[w[0],1]
                        listb[0] = bonds[w[0],0]
                        lista = np.array(lista)
                        listb = np.array(listb)
                        bonds[w[0],:] = -(nclust+1)
                        adda = 1
                        addb = 1
                        donea = 0
                        doneb = 0

                        if (donea != adda) | (doneb != addb):
                            true = 0
                        else:
                            true = 1

                        while true != 1:
                            # Expand through neighbors in bipartite graph
                            if (donea != adda):
                                w = np.where(bonds[:,1] == lista[donea])[0]
                                ngood = len(w)
                                if ngood != 0:
                                    listb[addb:addb+ngood] = bonds[w,0]
                                    bonds[w,:] = -(nclust+1)
                                    addb = addb+ngood
                                donea = donea + 1

                            if (doneb != addb):
                                w = np.where(bonds[:,0] == listb[doneb])[0]
                                ngood = len(w)
                                if ngood != 0:
                                    lista[adda:adda+ngood] = bonds[w,1]
                                    bonds[w,:] = -(nclust+1)
                                    adda = adda+ngood
                                doneb = doneb + 1

                            if (donea != adda) | (doneb != addb):
                                true = 0
                            else:
                                true = 1

                        # After expansion, compute cluster size
                        pqx = np.argsort(listb[0:doneb])
                        arr = listb[0:doneb]
                        q = arr[pqx]
                        indices = np.where(q != np.roll(q,-1))[0]
                        count = len(indices)
                        if count > 0:
                            unx = pqx[indices]
                        else:
                            unx = [len(q) - 1]
                        xsz = len(unx)

                        pqy = np.argsort(lista[0:donea])
                        arr = lista[0:donea]
                        q = arr[pqy]
                        indices = np.where(q != np.roll(q,-1))[0]
                        count = len(indices)
                        if count > 0:
                            uny = pqy[indices]
                        else:
                            uny = [len(q) - 1]
                        ysz = len(uny)

                        if xsz*ysz > maxsz:
                            maxsz = xsz*ysz

                        thru = thru - xsz
                        nclust = nclust + 1
                    bmap = bonds[:,1]   # Extract the second column of bonds (bonded partner indices)
                
                for nc in range(1,nclust+1):   # Loop over clusters from 1 to nclust

                    w = np.where(bmap == -1*(nc))[0]   # Find indices where bond partner = -nc (negative cluster label)

                    nbonds = len(w)   # Number of bonds in this cluster subset

                    bonds = mbonds[w,:]   # Extract subset of bonds corresponding to this cluster
                    lensq = bondlen[w]    # Corresponding bond lengths (squared distances)
                    st = np.argsort(bonds[:,0])   # Sort bonds by first column (node index)
                    arr = bonds[:,0]               # Extract first column (nodes)
                    q = arr[st]                    # Sorted nodes
                    indices = np.where(q != np.roll(q,-1))[0]   # Find places where consecutive entries differ
                    count = np.size(indices)
                    if count>0:
                        un = st[indices]   # Unique node indices from sorting
                    else:
                        un = np.size(q) - 1

                    uold = bonds[un,0].ravel()   # Unique set of old node indices (flattened)

                    nold = np.size(uold)   # Number of unique old nodes


                    indices = np.where(bonds[:,1] != np.roll(bonds[:,1],-1))[0]   # Unique in column 2
                    count = np.size(indices)

                    if count>0:
                        un = indices
                    else:
                        un = len(bonds[:,1]) - 1

                    unew = bonds[un,1]   # Unique new node indices
                    nnew = np.size(unew) # Number of new nodes

                    st = np.zeros(nnew,int)   # Start indices of chains
                    fi = np.zeros(nnew,int)   # End indices of chains
                    h = np.zeros(nbonds,int)  # Mapping bonds to old indices
                    ok = np.ones(nold)        # Boolean mask for available nodes
                    nlost = (nnew-nold)>0     # True if more new nodes than old (some are "lost")


                    for ii in range(nold):
                        index_ii = np.where(bonds[:,0] == uold[ii])[0]  # Find positions in bonds with this old node
                        h[index_ii] = ii   # Map them back to old index position

                    st[0] = 0             # First start index is 0
                    fi[nnew-1] = nbonds-1 # Last finish index is last bond

                    ##print(h)
                    bonds = bonds - 1     # Shift to 0-based indexing

                    if nnew>1:   # If there are at least 2 new nodes
                        sb = bonds[:,1]          # New node column
                        sbr = np.roll(sb,1)      # Right-shifted version
                        sbl = np.roll(sb,-1)     # Left-shifted version
                        st[1:len(st)] = np.where(sb[1:len(sb)] != sbr[1:len(sbr)])[0] + 1  # Where new index changes
                        fi[0:nnew-1] = np.where(sb[0:nbonds-1] != sbl[0:nbonds-1])[0]      # Where segment ends

                    checkflag = 0   # Control flag for outer while loop



                    while checkflag != 2:   # Continue until checkflag reaches 2

                        pt = st - 1               # Initialize pointers at start - 1
                        lost = np.zeros(nnew,int) # Track "lost" nodes
                        who = -1                  # Current node index under consideration
                        losttot = 0               # Total lost so far
                        mndisq = nnew*maxdisq     # Initialize best distance as large


                        while who != -2:   # Inner loop: assign nodes one by one

                            if pt[who+1] != fi[who+1]:   # If current pointer hasn’t reached the segment end

                                w = np.where(ok[h[pt[who+1]+1:fi[who+1]+1]])[0]  # Find valid next positions

                                ngood = len(w)   # Count good options

                                if ngood > 0:    # At least one good option
                                    if pt[who+1] != st[who+1] - 1:
                                        ok[h[pt[who+1]]] = 1   # Release previous assignment
                                    pt[who+1] = pt[who+1] + w[0] + 1  # Move pointer forward
                                    ok[h[pt[who+1]]] = 0   # Mark node as taken

                                    if who == nnew - 2:   # If last-but-one new node
                                        ww = np.where(lost == 0)[0]   # Indices of not lost
                                        dsq = sum(lensq[pt[ww]]) + losttot*maxdisq  # Total distance incl. penalty
                                        if dsq < mndisq:   # If better than current best
                                            minbonds = pt[ww]   # Store bonds
                                            mndisq = dsq
                                    else:
                                        who += 1   # Move to next new node
                                else:
                                    # No good options available
                                    if lost[who+1] == 0 and losttot != nlost:
                                        lost[who+1] = 1
                                        losttot += 1
                                        if pt[who+1] != st[who+1] - 1:
                                            ok[h[pt[who+1]]] = 1
                                        if who == nnew - 2:
                                            ww = np.where(lost == 0)[0]
                                            dsq = sum(lensq[pt[ww]])+losttot*maxdisq
                                            if dsq < mndisq:
                                                minbonds = pt[ww]
                                                mndisq = dsq
                                        else:
                                            who += 1
                                    else:   # Backtrack
                                        if pt[who+1] != (st[who+1] - 1):
                                            ok[h[pt[who+1]]] = 1
                                        pt[who+1] = st[who+1] - 1
                                        if lost[who+1] == 1:
                                            lost[who+1] = 0
                                            losttot = losttot - 1
                                        who = who - 1

                            else:  # If current pointer reached the end of segment
                                if lost[who+1] == 0 and losttot != nlost:
                                    lost[who+1] = 1
                                    losttot = losttot + 1
                                    if pt[who+1] != st[who+1] - 1:
                                        ok[h[pt[who+1]]] = 1
                                    if who == nnew-2:
                                        ww = np.where(lost == 0)[0]
                                        dsq  = sum(lensq[pt[ww]]) + losttot*maxdisq
                                        if dsq < mndisq:
                                            minbonds = pt[ww]
                                            mndisq = dsq
                                    else:
                                        who = who + 1
                                else:  # Backtrack again
                                    if pt[who+1] != st[who+1] - 1:
                                        ok[h[pt[who+1]]] = 1
                                    pt[who+1] = st[who+1] - 1
                                    if lost[who+1] == 1:
                                        lost[who+1] = 0
                                        losttot = losttot - 1
                                    who = who - 1

                        checkflag = checkflag + 1

                        if checkflag == 1:   # After first run, update nlost estimate
                            plost = min(np.fix(mndisq/maxdisq),(nnew-1))
                            if plost > nlost:
                                nlost = plost
                            else:
                                checkflag = 2   # Otherwise, exit outer loop

                    bonds = bonds.astype(int)  # Ensure integer indices
                    resx[ispan, labely[bonds[minbonds,1]]] = eyes[labelx[bonds[minbonds,0]+1]]
                    found[labelx[bonds[minbonds,0]+1]] = 1   # Mark this mapping as found


            # --------------------------
            # After assignment: update positions & bookkeeping
            # --------------------------
            w = np.where(resx[ispan,:] >= 0)[0]
            nww = len(w)

            resx = resx.astype(int)
            if nww > 0:
                pos[w,:] = xyzs[resx[ispan,w],0:dim]
                if goodenough > 0:
                    nvalid[w] = nvalid[w] + 2

            # Add new detections as new tracks if unlinked
            newguys = np.where(found == 0)[0]
            nnew = len(newguys)
            if nnew > 0:
                newarr = np.zeros([zspan,nnew]) - 1
                resx = np.hstack([resx,newarr])
                resx[ispan,np.arange(n,len(resx[0]))] = eyes[newguys]
                pos = np.vstack([pos,xyzs[eyes[newguys],0:dim]])
                nmem = np.zeros([nnew,1])
                mem = np.vstack([mem,nmem])
                nun = np.arange(0,nnew)
                uniqid = np.hstack([uniqid,(nun+maxid)])
                maxid += nnew

                if goodenough>0:
                    dumphashext = np.array(np.zeros([1,nnew])).T
                    dumphash = np.vstack([dumphash,dumphashext])
                    nvalidext = np.array(np.zeros([1,nnew])).T + 1
                    nvalid =  np.vstack([nvalid,nvalidext])

                n += nnew

        else:
            print('Warning: No detections in frames')

        # --------------------------
        # Update track memory
        # --------------------------
        w = np.where(resx[ispan,:] != -1)[0]
        nok = len(w)
        if nok != 0:
            mem[w] = 0
        mem = mem + (np.array([resx[ispan,:]]).T == -1)
        wlost = np.where(mem == memory_b + 1)[0]
        nlost = len(wlost)
        if nlost > 0:
            pos[wlost,:] = -maxdisp
            if goodenough > 0:
                wdump = np.where(nvalid[wlost] < goodenough)[0]
                ndump = len(wdump)
                if ndump > 0:
                    dumphash[wlost[wdump]] = 1

        # --------------------------
        # Block finalization: save results into bigresx, prune dead tracks
        # --------------------------
        if (ispan+1 == zspan) | (i+1 == z):
            Test.append(1)
            nold = len(bigresx[0,:])
            nnew = n - nold
            if nnew>0:
                newarr = np.zeros([z,nnew]) - 1
                bigresx = np.hstack([bigresx,newarr])

            if goodenough > 0:
                if sum(dumphash) > 0:
                    wkeep = np.where(dumphash == 0)[0]
                    nkeep = len(wkeep)
                    resx = resx[:,wkeep]
                    bigresx = bigresx[:,wkeep]
                    pos = pos[wkeep,:]
                    mem = mem[wkeep]
                    uniqid = uniqid[wkeep]
                    nvalid = nvalid[wkeep]
                    n = nkeep
                    dumphash = np.zeros([nkeep,1])

            bigresx[i-ispan:i+1,:] = resx[0:ispan+1,:]
            resx = np.zeros([zspan,n]) - 1

            # Handle lost tracks
            wpull = np.where(pos[:,0] == -maxdisp)[0]
            npull = len(wpull)
            if npull > 0:
                lillist = np.zeros([1,2])
                for ipull in range(0,npull):
                    wpull2 = np.where(bigresx[:,wpull[ipull]] != -1)[0]
                    npull2 = len(wpull2)
                    thing = np.hstack([np.array([bigresx[wpull2,wpull[ipull]]]).T,(np.zeros([npull2,1])+uniqid[wpull[ipull]])])
                    lillist = np.vstack([lillist,thing])
                olist = np.vstack([olist,lillist[1:len(lillist),:]])

            # Keep only active tracks
            wkeep = np.where(pos[:,0] >= 0)[0]
            nkeep = len(wkeep)
            resx = resx[:,wkeep]
            bigresx = bigresx[:,wkeep]
            pos = pos[wkeep,:]
            mem = mem[wkeep]
            uniqid = uniqid[wkeep]
            n = nkeep
            dumphash = np.zeros([nkeep,1])
            if goodenough > 0:
                nvalid = nvalid[wkeep]

    # --------------------------
    # Final cleanup: drop too-short tracks
    # --------------------------
    if goodenough > 0:
        nvalid = sum(bigresx >=0)
        wkeep = np.where(nvalid>=goodenough)[0]
        nkeep = len(wkeep)
        if nkeep < n:
            bigresx = bigresx[:,wkeep]
            n = nkeep
            pos = pos[wkeep,:]

    # --------------------------
    # Convert results into final track array
    # --------------------------
    wpull = np.where(pos[:,0] != -2*maxdisp)[0]
    npull = len(wpull)
    if npull>0:
        lillist = np.zeros([1,2])
        for ipull in range(npull):
            wpull2 = np.where(bigresx[:,wpull[ipull]] != -1)[0]
            npull2 = len(wpull2)
            thing = np.hstack([np.array([bigresx[wpull2,wpull[ipull]]]).T,(np.zeros([npull2,1])+uniqid[wpull[ipull]])])
            lillist = np.vstack([lillist,thing])
        olist = np.vstack([olist,lillist[1:len(lillist),:]])

    # Convert to numpy array
    olist = np.array(olist,int)
    olist = olist[1:len(olist)]

    nolist = len(olist[:,0])
    res = np.zeros([nolist,dd+2])

    # Copy linked data into result array
    for j in range(dd+1):
        res[:,j] = xyzs[olist[:,0],j]
    res[:,dd+1] = olist[:,1]

    ndat = len(res[0,:])
    newtracks = np.zeros(res.shape)
    newtracks[:,:] = res[:,:]

    # --------------------------
    # Assign new track IDs
    # --------------------------
    indices = np.where(newtracks[:,ndat-1] != np.roll(newtracks[:,ndat-1],-1))[0]
    count = len(indices)
    if count>0:
        u = indices
    else:
        u = np.array([len(newtracks[:,ndat-1]) - 1])
    ntracks = len(u)
    u = np.insert(u,0,-1)

    for i in range(1,ntracks+1):
        newtracks[u[i-1]+1:u[i]+1,ndat-1] = i
    
    return newtracks


def load(path):
    """
    Load a pickled object from a file.

    Parameters:
    - path: str, path to the .pkl file

    Returns:
    - data: object loaded from the pickle file
    """
    # Open a file in binary read mode
    dbfile = open(path, 'rb')

    # Load the serialized object using pickle
    data = pickle.load(dbfile)

    # Close the file to release resources
    dbfile.close()

    # Return the loaded object
    return data


def ismember(A, B):
    """
    Check membership of elements in A with respect to B.

    Parameters:
    - A: iterable of values to test
    - B: array to compare against

    Returns:
    - List of counts indicating how many times each element of A appears in B
    """
    return [np.sum(a == B) for a in A]


def create_horizontal_colorbar(min_val, max_val, width=250, height=30, colormap=cv.COLORMAP_JET):
    """
    Create a horizontal colorbar image using OpenCV colormap.

    Parameters:
    - min_val: minimum value represented by the colorbar
    - max_val: maximum value represented by the colorbar
    - width: int, width of the colorbar in pixels
    - height: int, height of the colorbar in pixels
    - colormap: OpenCV colormap identifier

    Returns:
    - colorbar: HxWx3 uint8 image representing the colorbar
    """
    gradient = np.linspace(max_val, min_val, width).astype(np.float32)
    norm = ((gradient - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    norm = norm.reshape((1, width))
    colorbar = cv.applyColorMap(norm, colormap)
    colorbar = cv.resize(colorbar, (width, height), interpolation=cv.INTER_NEAREST)
    return colorbar


def smooth_tracks(tracklist, rad, lim):
    """
    Smooth tracks using Savitzky-Golay filter if they meet displacement and length criteria.

    Parameters:
    - tracklist: list of 2D arrays, each array is a track [x, y, frame]
    - rad: minimum mean displacement threshold to apply smoothing
    - lim: minimum number of points in track to apply smoothing

    Returns:
    - smooth_tracks: list of smoothed tracks
    """
    numtracks = len(tracklist)
    smooth_tracks = []
    for i in range(numtracks):
        track = tracklist[i]
        if len(track) > lim:
            if np.linalg.norm(np.mean(track[:, 0:2], axis=0) - track[0, 0:2]) > rad:
                tracksmooth = np.zeros_like(track)
                smoother = int(min(51, np.ceil(len(track)/2.0)*2 - 1))
                poly = min(3, int(smoother/3))
                tracksmooth[:, 0] = savgol_filter(track[:, 0], smoother, poly)
                tracksmooth[:, 1] = savgol_filter(track[:, 1], smoother, poly)
                tracksmooth[:, 2] = np.array(track[:, 2], int)
                smooth_tracks.append(tracksmooth)
    return smooth_tracks


def make_title_with_colorbar(px, min_val, max_val, total_width, colorbar_width=250, height=60, num_ticks=6, title="Color bar represents Track duration [in frames]"):
    """
    Generate an image row containing a title text and horizontal colorbar with labels.

    Parameters:
    - px: int, background pixel intensity
    - min_val: minimum value for colorbar
    - max_val: maximum value for colorbar
    - total_width: int, width of the title row
    - colorbar_width: width of colorbar
    - height: height of title row
    - num_ticks: number of tick labels on colorbar
    - title: string, title text to display

    Returns:
    - title_row: HxWx3 uint8 image containing title and colorbar
    """
    colorbar_height = int(height/2)
    label_area_height = height

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    title_row = np.ones((label_area_height, total_width, 3), dtype=np.uint8) * px
    text_x = int(0.5*total_width - colorbar_width)
    text_y = int(label_area_height / 2 + height/6)
    cv.putText(title_row, title, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv.LINE_AA)

    colorbar = create_horizontal_colorbar(min_val, max_val, width=colorbar_width, height=colorbar_height)
    label_bar = np.ones((label_area_height, colorbar_width, 3), dtype=np.uint8) * px
    label_bar[-colorbar_height:] = colorbar

    for i in range(num_ticks):
        x = min(int(i * colorbar_width / (num_ticks - 1)), colorbar_width - 1)
        val = min_val + i * (max_val - min_val) / (num_ticks - 1)
        label = f"{val:.0f}"
        color = tuple(int(c) for c in colorbar[-1, x])
        text_size = cv.getTextSize(label, font, 0.5, 1)[0]

        if i == 0:
            text_x = max(x, 0)
        elif i == num_ticks - 1:
            text_x = min(x - text_size[0], colorbar_width - text_size[0])
        else:
            text_x = x - text_size[0] // 2

        text_y = label_area_height - colorbar_height - 5
        cv.putText(label_bar, label, (text_x, text_y), font, 0.5, color, 1, cv.LINE_AA)

    bar_start_x = total_width - 2*colorbar_width
    title_row[:, bar_start_x:bar_start_x+colorbar_width] = label_bar
    return title_row



def image_with_tracks(image, tracklist, respath, keep_tracks=True, lw=2, show_cbar=False):
    """
    Overlay tracks on an image stack and optionally add a colorbar for track duration.

    Parameters:
    - image: list or array of image frames (HxWx3)
    - tracklist: list of tracks, each track is an array [x, y, frame]
    - respath: string, path to save output image stack
    - keep_tracks: bool, whether to persist all tracks over frames
    - lw: int, line width for track drawing
    - show_cbar: bool, whether to overlay a colorbar indicating track durations

    Returns:
    - None (writes output to respath and displays last frame using plt.imshow)
    """
    px = int(np.mean(image[0,:,:,0]))
    numtracks = len(tracklist)
    tracklengths = np.zeros(numtracks)
    fac = 255

    for t in range(numtracks):  
        tracklengths[t] = len(tracklist[t])

    min_len = min(tracklengths)
    max_len = max(tracklengths)
    norm_lengths = [(l - min_len) / (max_len - min_len) if max_len > min_len else 0 for l in tracklengths]
    cmap = plt.cm.jet
    bgr_colors = [tuple(int(fac * c) for c in cmap(n)[:3]) for n in norm_lengths]

    for i in tqdm(range(numtracks), desc='Plotting tracks over image:'):
        track = tracklist[i]
        Color = bgr_colors[i]
        data = track[:,0:2]
        x, y = data.T
        x = np.array(x,int)
        y = np.array(y,int)
        start_frame = int(track[:,2][0]-1)
        counter = 0

        if keep_tracks:
            keep_ind = len(image)+1
        else:
            keep_ind = start_frame+len(x)+1

        while (start_frame+counter) < keep_ind-1:
            extent = min(counter,len(x)-1)
            for z in range(1,extent):
                if counter == 0:
                    image[start_frame] = cv.circle(image[start_frame], (x[0],y[0]), radius=0, color=Color, thickness=lw)
                else:
                    image[start_frame+counter] = cv.line(image[start_frame+counter], (x[z-1],y[z-1]), (x[z],y[z]), color=Color, thickness=lw)
            counter += 1

    if show_cbar:
        image_cbar = []
        for i,img in enumerate(image):
            bar_width = img.shape[1]
            colorbar_with_labels = make_title_with_colorbar(px, min_len, max_len, bar_width, height=int(0.02*len(img)))
            final_output = np.vstack((colorbar_with_labels, img))
            image_cbar.append(final_output)
        image_cbar = np.array(image_cbar,dtype=np.uint8)
        tf.imwrite(respath,image_cbar)
        plt.imshow(image_cbar[-1])
    else:
        tf.imwrite(respath,image)
        plt.imshow(image[-1])


def plot_tracks_with_background(tracklist, image, path='none', show_ind=True):
    """
    Plot all tracks on a figure showing both centered trajectories and original image background.

    Parameters:
    - tracklist: list of tracks, each track is an array [x, y, frame]
    - image: 2D array, grayscale background image
    - path: str, optional path to save the figure
    - show_ind: bool, whether to label track indices on the plot

    Returns:
    - None (optionally saves figure to disk)
    """
    numtracks = len(tracklist)
    tracklengths = np.zeros(numtracks)
    tracklist_origin = []

    for t in range(numtracks):
        track = tracklist[t][:,0:2]
        track = track - track[0]
        tracklengths[t] = len(track)
        tracklist_origin.append(track)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(19,9))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_position(('data',0))
    ax1.spines['left'].set_position(('data',0))

    min_len = min(tracklengths)
    max_len = max(tracklengths)
    norm = [(l - min_len) / (max_len - min_len) if max_len > min_len else 0 for l in tracklengths]
    cmap = plt.cm.jet
    colors = [cmap(n) for n in norm]

    j = 0
    for traj, length in zip(tracklist_origin, tracklengths):
        color = colors[j]
        alp = min(length/100, 1)
        ax1.plot(traj[:, 0], traj[:, 1], color=color, lw=1, alpha=alp)
        j += 1

    ax2.imshow(image,cmap='gray')
    i = 0
    for traj, length in zip(tracklist, tracklengths):
        color = colors[i]
        alp = min(length/100, 1)
        ax2.plot(traj[:, 0], traj[:, 1], color=color, lw=1, alpha=alp)
        if show_ind:
            mid_pt_x = traj[:,0][int(0.5*len(traj))] + 5
            mid_pt_y = traj[:,1][int(0.5*len(traj))] + 5
            ax2.text(mid_pt_x, mid_pt_y, f'{i}', color=color, fontsize=10)
        i += 1

    norm = Normalize(vmin=min(tracklengths), vmax=max(tracklengths))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1)
    plt.subplots_adjust(left=0.01, right=0.98, top=0.99, bottom=0.05, wspace=0.2)
    ax2.text(-0.25, 0.5, 'Colorbar: Duration of tracks in frames', va='center', transform=ax2.transAxes, rotation=-90, fontsize=18)

    if path != 'none':
        plt.savefig(path, facecolor='white', dpi=500)


def evaluate_angle(A, B):
    """
    Compute signed angle between two 2D vectors using arctangent of determinant and dot product.

    Parameters:
    - A: array-like, first 2D vector
    - B: array-like, second 2D vector

    Returns:
    - angle: float, angle in radians between A and B
    """
    dot = np.dot(A,B)
    det = A[0]*B[1] - A[1]*B[0]
    return np.arctan2(det,dot)


def filter_tracks(tracklist, speed_lim=0.77, length_llim=10, length_ulim=400, disp_llim=30, curve_lim=0.8):
    """
    Filter tracks based on speed, length, displacement, and curvature thresholds.

    Parameters:
    - tracklist: list of tracks, each track is an array [x, y, frame]
    - speed_lim: float, minimum average speed to keep track
    - length_llim: int, minimum track length in frames
    - length_ulim: int, maximum track length in frames
    - disp_llim: float, minimum displacement from start to end
    - curve_lim: float, maximum median curvature allowed

    Returns:
    - filtered_tracklist: list of tracks that satisfy all criteria
    """
    filtered_tracklist = []

    for i in range(len(tracklist)):
        data = tracklist[i]
        track = data[:,0:2]

        if not np.isnan(np.sum(track[:,0])):
            velocity_cartesian = np.ones_like(track) * float('NaN')
            velocity_polar = np.ones_like(track) * float('NaN')
            velocity_cartesian[:-1] = np.diff(track, axis=0)
            velocity_r = np.linalg.norm(velocity_cartesian[:-1], axis=1)
            velocity_theta = np.arctan2(velocity_cartesian[:-1][:,1], velocity_cartesian[:-1][:,0])
            velocity_polar[:-1] = np.column_stack([velocity_r, velocity_theta])

            turnrate = np.ones(len(track)) * float('NaN')
            turns = [np.round(evaluate_angle(velocity_cartesian[k], velocity_cartesian[k+1]),4) for k in range(len(velocity_cartesian)-2)]
            turnrate[1:-1] = turns

            pathlength = np.zeros(len(track))
            pathlength[1:] = np.nancumsum(velocity_r)

            curvature = np.ones(len(track)) * float('NaN')
            diff_angles = (np.array(turns) - np.pi + np.pi) % (2 * np.pi) - np.pi
            curvature[1:-1] = diff_angles / velocity_r[:-1]

            length = np.ones(len(track)) * float('NaN')
            length[0] = len(track)

            distance = np.ones(len(track)) * float('NaN')
            distance[0] = pathlength[-1]

            median_curvature = np.ones(len(track)) * float('NaN')
            median_curvature[0] = np.nanmedian(abs(curvature))

            speed = np.ones(len(track)) * float('NaN')
            speed[0] = pathlength[-1] / (len(track)-1)

            displacement = np.ones(len(track)) * float('NaN')
            displacement[0] = np.linalg.norm(track[-1]-track[0])

            if speed[0] > speed_lim and length_llim < length[0] < length_ulim and displacement[0] > disp_llim and 0 < median_curvature[0] < curve_lim:
                filtered_tracklist.append(data)

    return filtered_tracklist


def save_track_csv(tracklist, parameter_dict, path):
    """
    Save each track as a CSV file and write a summary text file of analysis parameters.

    Parameters:
    - tracklist: list of tracks, each track is an array [x, y, frame]
    - parameter_dict: dictionary of parameters used for analysis
    - path: string, directory to save CSV files and summary

    Returns:
    - None (writes CSV files to disk and creates statistics_summary.txt)
    """
    tracks_dir = os.path.join(path, 'saved_tracks')
    if not os.path.exists(tracks_dir):
        os.mkdir(tracks_dir)
    else:
        shutil.rmtree(tracks_dir)
        os.mkdir(tracks_dir)

    for i in range(len(tracklist)):
        track = tracklist[i]
        df = pd.DataFrame(track)
        df.columns = ['X','Y','Frame']
        filename = os.path.join(tracks_dir, f'track_{i}.csv')
        df.to_csv(filename)

    save_lines = []
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_lines.append(f'This file was generated by TrackScoPy: main_image_analysis.ipynb on {current_datetime}')
    save_lines.append('')
    save_lines.append('==============================================================================================================')
    save_lines.append('')
    save_lines.append(f'Total number of tracks analyzed: {len(tracklist)}')
    save_lines.append('')
    save_lines.append('==============================================================================================================')
    save_lines.append('')
    save_lines.append('Parameters used for analysis:')
    save_lines.append('')
    save_lines.append('')
    sum_path = os.path.join(path, 'statistics_summary.txt')
    with open(sum_path, 'w') as f:
        f.write('\n'.join(save_lines))
        for key, value in parameter_dict.items():
            f.write(f"{key} = {value}\n")

