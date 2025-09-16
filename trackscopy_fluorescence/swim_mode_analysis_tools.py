# -*- coding: utf-8 -*-
"""
Spyder Editor

- This is a module to process trajectories from separate color channels of fluorescence images 
  and classify bacteria based on their swim-modes. This analysis framework is very specific to motility experiments
  for the soil bacterium Pseudomonas putida under fluorescence microscopy. 

- Author: Agniva Datta @ University of Potsdam, Germany


"""

# =======================
# Import Required Modules
# =======================

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import tifffile as tf
import cv2 as cv
import pickle
import shutil
from datetime import datetime
import os

# ======================================================
# Load default parameters from external parameter script
# ======================================================

with open('trackscopy_fluorescence/default_parameters_swim_mode_detection.py') as f:
    exec(f.read())
    
# =========================
# Swim Mode Analysis Class
# =========================

class Swim_Mode_Detection():
    """
    Class to detect swimming modes of bacteria (push, pull, wrap, passive)
    based on dual-channel (red: cell body, green: flagella) tracking data.
    """

    def __init__(self,red_channel,green_channel,parameters=default_parameters_swim_mode_detection,calibration=None):
        """
        Initialize the Swim_Mode_Detection object.
        
        Args:
            red_channel: processed red channel object (contains segmented images & tracks of cell bodies).
            green_channel: processed green channel object (contains segmented images & tracks of flagella).
            parameters: list of swim-mode detection parameters (ARROW_LENGTH, ARROW_LIFETIME, BAC_SIZE, BACKGROUND_FACTOR).
            calibration: optional calibration values for velocity normalization and scaling.
        """

        # Store channels
        self.red_channel = red_channel
        self.green_channel = green_channel

        # Extract image dimensions
        self.nImg = len(self.red_channel.segmented_image.im_array)   # number of frames
        self.ht = len(self.red_channel.segmented_image.im_array[0])  # image height
        self.wt = len(self.red_channel.segmented_image.im_array[0][0])  # image width

        # Unpack parameters
        self.ARROW_LENGTH,self.ARROW_LIFETIME, self.BAC_SIZE, self.BACKGROUND_FACTOR = parameters

        # Calibration setup: motion vector and standard deviation
        if calibration is None:
            self.mV = np.array([0,0])  # mean velocity
            self.mS = np.array([0,0])  # spread
            self.mV_abs = 0            # magnitude of mean velocity
        else:
            self.mV = calibration[0:2]
            self.mS = calibration[2:4]
            self.mV_abs = np.linalg.norm(self.mV)

    def Analysis(self):
        """
        Core routine: match red and green tracks, compute relative positions, 
        detect swim modes, calculate orientation and flow alignment.
        Produces a dataframe with swim-mode labeled trajectories.
        """

        # Count tracks in both channels
        numtracks1 = len(self.red_channel.smooth_tracks)
        numtracks2 = len(self.green_channel.smooth_tracks)
        
        # Define border cutoffs to exclude detections near edges
        borderx = int(self.BACKGROUND_FACTOR*self.wt)
        bordery = int(self.BACKGROUND_FACTOR*self.ht)

        # ============================
        # Build red track data
        # ============================
        red_track = np.array([0.0,0.0,0.0,0.0])
        for i in range(numtracks1):
            r_track_red = self.red_channel.smooth_tracks[i]
            ID_column = int(i+1)*np.ones(len(r_track_red))  # unique ID for each trajectory
            r_track_red_id = np.column_stack((r_track_red,ID_column))
            red_track = np.vstack([red_track,r_track_red_id])
        red_track = np.delete(red_track,0,0)  # remove placeholder row
    
        # ============================
        # Build green track data
        # ============================
        green_track = np.array([0.0,0.0,0.0,0.0])
        for i in range(numtracks2):
            g_track_green = self.green_channel.smooth_tracks[i]
            ID_column = int(i+1)*np.ones(len(g_track_green)) 
            g_track_green_id = np.column_stack((g_track_green,ID_column))
            green_track = np.vstack([green_track,g_track_green_id])
        green_track = np.delete(green_track,0,0)
    
        # ============================
        # Compute velocities for red tracks
        # ============================
        red_track_vel = np.array([0.0,0.0])
        for i in range(numtracks1):
            r_track_red = self.red_channel.smooth_tracks[i]
            r_track_vel = np.zeros_like(r_track_red[:,0:2])  # initialize velocity array
            for k in range(1,len(r_track_red)):
                dx1 = r_track_red[k,0:2] - r_track_red[k-1,0:2]
                dt1 = r_track_red[k,-1] - r_track_red[k-1,-1]
                r_track_vel[k,:] = dx1/dt1   # velocity = displacement / time
            r_track_vel[0] = r_track_vel[1]  # copy second velocity to first index
            red_track_vel = np.vstack([red_track_vel,r_track_vel])
        red_track_vel = np.delete(red_track_vel,0,0)

        # Initialize lists to store computed quantities
        test_time_red = []
        test_time_green = []
        Mode, Sign, Size, Pointer = [], [], [], []
        cellbody, flagella, time_cellbody, Com = [], [], [], []
        velocity, speed = [], []
        red_id, green_id = [], []

        # ============================
        # Iterate over frames and try to match red (cellbody) and green (flagella) detections
        # ============================
        for t in range(self.nImg):

            # Frame-wise storage
            cX1, cX2, cY1, cY2, vX1, vY1 = [], [], [], [], [], []
            T1, T2, T11, T22, ID1, ID2 = [], [], [], [], [], []

            # Gather red detections at frame t
            for i in range(len(red_track)):
                if t == int(red_track[i,2]):
                    cX1.append(red_track[i,0])
                    cY1.append(red_track[i,1])
                    vX1.append(red_track_vel[i,0])
                    vY1.append(red_track_vel[i,1])
                    T1.append(red_track[i,2]-red_track[0,2])
                    T11.append(red_track[i,2])
                    ID1.append(red_track[i,3])
                    test_time_red.append(red_track[i,2])

            # Gather green detections at frame t
            for j in range(len(green_track)):       
                if t == int(green_track[j,2]):
                    cX2.append(green_track[j,0])
                    cY2.append(green_track[j,1])
                    T2.append(green_track[j,2]-green_track[0,2])
                    T22.append(green_track[j,2])
                    ID2.append(green_track[j,3])
                    test_time_green.append(green_track[j,2])

            # Matching criteria
            dx = self.ARROW_LENGTH   # max allowed spatial separation
            dt = self.ARROW_LIFETIME # max allowed temporal separation

            Headx, Tailx, Heady, Taily = [], [], [], []

            # ============================
            # Attempt pairing of red and green detections
            # ============================
            for k in range(max(len(cX1),len(cX2))):
                if len(cX1) >= len(cX2):
                    # Loop over green points
                    for l in range(len(cX2)):
                        # Spatial & temporal proximity
                        if np.sqrt((cX1[k]-cX2[l])**2 + (cY1[k]-cY2[l])**2) < dx:
                            if abs(T1[k] - T2[l]) < dt:
                                # Ensure detection is not near the border
                                if borderx < cX2[l] < (self.wt - borderx) and bordery < cY2[l] < (self.ht - bordery):
                                    # Prevent duplicate matches
                                    if len(np.where(np.array(Headx) == cX1[k])[0]) == 0 and len(np.where(np.array(Heady) == cY1[k])[0]) == 0:
                                        if len(np.where(np.array(Tailx) == cX2[l])[0]) == 0 and len(np.where(np.array(Taily) == cY2[l])[0]) == 0:
                                            # Save matched pair
                                            pointer = np.array([(cX1[k]-cX2[l]),cY1[k]-cY2[l]],float)
                                            com = 0.5*np.array([(cX1[k]+cX2[l]),cY1[k]+cY2[l]],float)
                                            X = np.array([cX1[k],cY1[k]])  # cell body pos
                                            F = np.array([cX2[l],cY2[l]])  # flagella pos
                                            V = np.array([vX1[k],vY1[k]])  # velocity
                                            cellbody.append(X)
                                            flagella.append(F)
                                            velocity.append(V)
                                            speed.append(np.linalg.norm(V))
                                            time_cellbody.append(T11[k])
                                            Pointer.append(pointer)
                                            Com.append(com)
                                            Size.append(np.linalg.norm(pointer))
                                            Headx.append(cX1[k]); Heady.append(cY1[k])
                                            Tailx.append(cX2[l]); Taily.append(cY2[l])
                                            red_id.append(ID1[k]); green_id.append(ID2[l])

                else:
                    # Symmetric case: fewer red than green
                    for l in range(len(cX1)):
                        if np.sqrt((cX1[l]-cX2[k])**2 + (cY1[l]-cY2[k])**2) < dx:
                            if abs(T1[l]-T2[k]) < dt:                        
                                if cX2[k] > borderx and cY2[k] > bordery:
                                    if len(np.where(np.array(Tailx) == cX2[k])[0]) == 0 and len(np.where(np.array(Taily) == int(cY2[k]))[0]) == 0:
                                        if len(np.where(np.array(Headx) == cX1[l])[0]) == 0 and len(np.where(np.array(Heady) == cY1[l])[0]) == 0:
                                            com = 0.5*np.array([(cX1[l]+cX2[k]),cY1[l]+cY2[k]],float)
                                            pointer = np.array([(cX1[l]-cX2[k]),cY1[l]-cY2[k]],float)
                                            X = np.array([cX1[l],cY1[l]])
                                            F = np.array([cX2[k],cY2[k]])
                                            V = np.array([vX1[l],vY1[l]])
                                            cellbody.append(X)
                                            flagella.append(F)
                                            velocity.append(V)
                                            speed.append(np.linalg.norm(V))
                                            time_cellbody.append(T11[l])
                                            Pointer.append(pointer)
                                            Com.append(com)
                                            Size.append(np.linalg.norm(pointer))
                                            Headx.append(cX1[l]); Heady.append(cY1[l])
                                            Tailx.append(cX2[k]); Taily.append(cY2[k])
                                            red_id.append(ID1[l]); green_id.append(ID2[k])

        # ============================
        # Classify swimming modes
        # ============================
        Sign, Orientation_Angle, Flow_Angle, Mode = [], [], [], []
        Microns_per_sec = np.zeros(len(velocity),float)

        for i in range(len(velocity)): 
            V = velocity[i]
            V_res = np.linalg.norm(V-self.mV)              # velocity relative to mean
            sign = np.dot(Pointer[i],(velocity[i]-self.mV)) # dot product for push/pull direction

            # Orientation angle relative to calibration vector
            if self.mV_abs != 0:                
                orientation_angle = (180/(np.pi))*evaluate_angle(self.mV,Pointer[i])
            else:
                orientation_angle = (180/(np.pi))*evaluate_angle(np.array([0,1]),Pointer[i])

            # Flow alignment angle
            flow_angle = (180/(np.pi))*evaluate_angle(self.mV,(velocity[i]-self.mV))

            # Classification logic
            if self.mV[0]-self.mS[0] < V[0] < self.mV[0]+self.mS[0] and self.mV[1]-self.mS[1] < V[1] < self.mV[1]+self.mS[1]:
                # Passive
                Mode.append(0.5)
                Microns_per_sec[i] = float('nan')
                Orientation_Angle.append(orientation_angle)
                Flow_Angle.append(float('nan'))
            else:
                Sign.append(sign)
                size = Size[i]
                if size<self.BAC_SIZE:
                    # Wrap mode (flagella bent around cell body)
                    Mode.append(-1)
                    Orientation_Angle.append(float('nan'))
                    Flow_Angle.append(flow_angle)
                else:
                    # Push or Pull depending on sign
                    if sign>0:
                        Mode.append(1)   # push
                    else:
                        Mode.append(0)   # pull
                    Orientation_Angle.append(orientation_angle)
                    Flow_Angle.append(flow_angle)

                # Velocity in microns/sec
                Microns_per_sec[i] = V_res*self.red_channel.framerate/self.red_channel.resolution

        # ============================
        # Store orientation & flow categories
        # ============================
        orientation_angle_dist, flow_angle_dist = [], []
        for i in range(len(Mode)):
            orientation_angle_dist.append(Orientation_Angle[i] if Mode[i]!=-1 else float('nan'))
            flow_angle_dist.append(Flow_Angle[i] if Mode[i]!=0.5 else float('nan'))

        # Orientation: parallel / perpendicular to mean flow
        Orientation = []
        for i in range(len(Orientation_Angle)):
            if Mode[i] != -1:
                if 45<Orientation_Angle[i]<135 or -135<Orientation_Angle[i]<-45:
                    Orientation.append('perpendicular')
                else:
                    Orientation.append('parallel')
            else:
                Orientation.append('NA')

        # Flow: up / down / NA
        Flow = []
        for i in range(len(Flow_Angle)):
            if Flow_Angle[i] == 0:
                Flow.append('NA')
            else:
                if Mode[i] != 0.5:
                    if -90<Flow_Angle[i]<90:
                        Flow.append('down')
                    else:
                        Flow.append('up')
                else:
                    Flow.append('NA')

        # Final swim mode labels
        Mode = np.array(Mode,float)
        Modes = []
        for i in range(len(Mode)):
            if Mode[i] == 1:
                Modes.append('push')
            elif Mode[i] == 0:
                Modes.append('pull')
            elif Mode[i] == -1:
                Modes.append('wrap')
            else:
                Modes.append('passive')

        # ============================
        # Construct DataFrame with results
        # ============================
        frames = [int(x+1) for x in time_cellbody]
        c_x = [round(x[0],2) for x in cellbody]
        c_y = [round(x[1],2) for x in cellbody]
        f_x = [round(x[0],2) for x in flagella]
        f_y = [round(x[1],2) for x in flagella]
        r_id = [int(x) for x in red_id]
        g_id = [int(x) for x in green_id]
        c_v = [round(x,2) for x in  Microns_per_sec]
        or_ang = [round(x,2) for x in orientation_angle_dist]
        fl_ang = [round(x,2) for x in flow_angle_dist]

        # Main results DataFrame
        self.data_with_id = pd.DataFrame({
            'FRAME': frames,'SWIM-MODE': Modes,'C:X-COORD': c_x, 'C:Y-COORD': c_y,
            'F:X-COORD': f_x, 'F:Y-COORD': f_y,'VELOCITY': c_v, 'ORIENTATION': Orientation, 
            'FLOW-DIRECTION':Flow, 'ORIENTATION-ANGLE': or_ang, 
            'FLOW-ANGLE': fl_ang,'CELL-ID':r_id, 'FLAGELLA-ID':g_id
        })

        # Post-processing: organize data, plot, and extract swimmer objects
        self.sort_data()
        self.plot_tracks()
        self.extract_swimmers()
        self.save_parameters()
        
    def sort_data(self,sort_by='CELL-ID'):
        """
        Sort the main dataset by a given identifier.

        Parameters:
        - sort_by: column name to sort by (default 'CELL-ID').

        After execution:
        - self.sorted_data contains a list of DataFrames, one per unique ID.
        - self.sorted_by stores the field used for sorting.
        """
        df = self.data_with_id
        unique_ids = df[sort_by].unique()
        self.sorted_data = [df[df[sort_by] == uid] for uid in unique_ids]
        self.sorted_by = sort_by

    def plot_tracks(self,path=None):
        """
        Plot trajectories and velocity traces for all sorted tracks.

        Parameters:
        - path: optional path to save the figure. If None, figure is not saved.

        Axes:
        - ax1: X-Y coordinates colored by track.
        - ax2: Frame vs velocity plot.
        """
        dfs = self.sorted_data
        Alpha = np.linspace(0.5,1,len(dfs))

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

        for i,df in enumerate(dfs):
            X = np.array(df['C:X-COORD'],float)
            Y = np.array(df['C:Y-COORD'],float)
            modes = np.array(df['SWIM-MODE'],str)
            velocities = np.array(df['VELOCITY'],float)
            frames = np.array(df['FRAME'],int)

            # Identify indices by swimming mode
            pushers = np.where(modes=='push')[0]
            pullers = np.where(modes=='pull')[0]
            wrappers = np.where(modes == 'wrap')[0]
            passive = np.where(modes == 'passive')[0]

            # Plot each mode
            if len(pushers) != 0:
                ax1.plot(X[pushers],Y[pushers],'b.',alpha=Alpha[i],markersize=8)
                ax1.text(np.mean(X[pushers])+5,np.mean(Y[pushers])+5,f'{i}')
                ax2.plot(frames[pushers],velocities[pushers],'b.',alpha=Alpha[i],markersize=8)
            if len(pullers) != 0:
                ax1.plot(X[pullers],Y[pullers],'r.',alpha=Alpha[i],markersize=8)
                ax1.text(np.mean(X[pullers])+5,np.mean(Y[pullers])+5,f'{i}')
                ax2.plot(frames[pullers],velocities[pullers],'r.',alpha=Alpha[i],markersize=8)
            if len(wrappers) != 0:
                ax1.plot(X[wrappers],Y[wrappers],'g.',alpha=Alpha[i],markersize=8)
                ax1.text(np.mean(X[wrappers])+5,np.mean(Y[wrappers])+5,f'{i}')
                ax2.plot(frames[wrappers],velocities[wrappers],'g.',alpha=Alpha[i],markersize=8)
            if len(passive) != 0:
                ax1.plot(X[passive],Y[passive],'k.',alpha=Alpha[i],markersize=8)
                ax1.text(np.mean(X[passive])+5,np.mean(Y[passive])+5,f'{i}')
                ax2.plot(frames[passive],velocities[passive],'k.',alpha=Alpha[i],markersize=8)

        ax1.set_xlabel(r'X-coordinate in $\mu m$',fontsize=12)
        ax1.set_ylabel(r'Y-coordinate in $\mu m$',fontsize=12)
        ax2.set_xlabel('Frames',fontsize=12)
        ax2.set_ylabel(r'Velocity in $\mu m/s$',fontsize=12)

        if path is None:
            pass
        else:
            plt.savefig(path,facecolor='white')

    def extract_swimmers(self):
        """
        Convert sorted DataFrames into Swimmer objects for further analysis.

        After execution:
        - self.swimmers contains a list of Swimmer objects.
        """
        self.swimmers = [Swimmer(x) for x in self.sorted_data]

    def write_swim_image(self,path,offset=20,show_velocity=False,show_orientation=False):
        """
        Overlay swim mode, velocity, and orientation on combined red/green channel images.

        Parameters:
        - path: output file path to save the image stack.
        - offset: pixel offset for text placement around swimmer center of mass.
        - show_velocity: if True, overlay velocity next to swimmer.
        - show_orientation: if True, overlay orientation label next to swimmer.

        Workflow:
        - Collect all swimmer positions, modes, velocities, and orientations.
        - Merge red and green channel images.
        - Loop over frames and overlay text for each swimmer present in that frame.
        """
        time_cellbody = []
        modes = []
        Com = []
        velocities = []
        orientations = []

        for i,swimmer in enumerate(self.swimmers):
            df = swimmer.dataframe
            time_cellbody.extend(df['FRAME'].to_numpy(dtype=int))
            modes.extend(df['SWIM-MODE'].to_numpy(dtype=str))
            C = df[['C:X-COORD', 'C:Y-COORD']].to_numpy(dtype=float)
            F = df[['F:X-COORD', 'F:Y-COORD']].to_numpy(dtype=float)
            velocities.extend(df['VELOCITY'].to_numpy(dtype=float))
            orientations.extend(df['ORIENTATION'].to_numpy(dtype=str))
            Com.extend(0.5*(C + F))  # center-of-mass positions

        Swim_img = combine_channel(self.red_channel.segmented_image.im_array,self.green_channel.segmented_image.im_array)
        nImg = len(Swim_img)

        offset = 20
        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5

        # Define colors for different swim modes
        color_wrap = (0,255,0)
        color_pull = (255,0,0)
        color_push = (0,0,255)
        color_passive = (0,0,0)
        thickness = 1

        # Draw labels on each frame
        for t in range(nImg):
            for i in range(len(modes)):
                if int(time_cellbody[i]) == t+1:
                    org = (int(Com[i][0]-offset),int(Com[i][1]-offset))
                    org1 = (int(Com[i][0]-2*offset),int(Com[i][1]-2*offset))
                    org2 = (int(Com[i][0]+2*offset),int(Com[i][1]+2*offset))

                    if modes[i] == 'wrap':
                        cv.putText(Swim_img[t], 'Wrap', org, font, fontScale, color_wrap, thickness, cv.LINE_AA)
                        if show_velocity: cv.putText(Swim_img[t], rf'{velocities[i]}$mu/s$', org1, font, fontScale, color_wrap, thickness, cv.LINE_AA)
                        if show_orientation: cv.putText(Swim_img[t],f'{orientations[i]}', org2, font, fontScale, color_wrap, thickness, cv.LINE_AA)
                    elif modes[i] == 'pull':
                        cv.putText(Swim_img[t], 'Pull', org, font, fontScale, color_pull, thickness, cv.LINE_AA)
                        if show_velocity: cv.putText(Swim_img[t], rf'{velocities[i]}$mu/s$', org1, font, fontScale, color_pull, thickness, cv.LINE_AA)
                        if show_orientation: cv.putText(Swim_img[t],f'{orientations[i]}', org2, font, fontScale, color_pull, thickness, cv.LINE_AA)
                    elif modes[i] == 'push':
                        cv.putText(Swim_img[t], 'Push', org, font, fontScale, color_push, thickness, cv.LINE_AA)
                        if show_velocity: cv.putText(Swim_img[t], rf'{velocities[i]}$mu/s$', org1, font, fontScale, color_push, thickness, cv.LINE_AA)
                        if show_orientation: cv.putText(Swim_img[t],f'{orientations[i]}', org2, font, fontScale, color_push, thickness, cv.LINE_AA)
                    elif modes[i] == 'passive':
                        cv.putText(Swim_img[t], 'Passive', org, font, fontScale, color_passive, thickness, cv.LINE_AA)
                        if show_orientation: cv.putText(Swim_img[t],f'{orientations[i]}', org2, font, fontScale, color_passive, thickness, cv.LINE_AA)

        tf.imwrite(path,Swim_img)  # save the final annotated image stack

    def write_arrow_image(self,path):
        """
        Generate and save an arrow image stack representing flagella-to-cellbody vectors.

        Parameters:
        - path: output path to save the arrow image stack.

        Workflow:
        - Collect frame-wise positions of cell bodies (C) and flagella tips (F) from all swimmers.
        - Draw arrows from F to C for each swimmer present in each frame.
        - Save inverted grayscale image stack.
        """
        time_cellbody = []
        C = []
        F = []

        # Collect all frame indices and coordinates
        for i,swimmer in enumerate(self.swimmers):
            df = swimmer.dataframe
            time_cellbody.extend(df['FRAME'].to_numpy(dtype=int))
            C.extend(np.array(df[['C:X-COORD', 'C:Y-COORD']]))
            F.extend(np.array(df[['F:X-COORD', 'F:Y-COORD']]))

        Arrow_img = np.zeros(self.red_channel.segmented_image.im_array.shape,dtype='uint8')

        # Draw arrows frame by frame
        for t in range(self.nImg):
            for i in range(len(time_cellbody)):
                if int(time_cellbody[i]) == t+1:
                    cv.arrowedLine(Arrow_img[t], (int(F[i][0]),int(F[i][1])), (int(C[i][0]),int(C[i][1])), color = 255, thickness = 2)

        tf.imwrite(path,(255-Arrow_img))  # save inverted arrow images

    def save_parameters(self):
        """
        Save current analysis parameters into self.parameter_dict for reference.
        """
        self.parameter_dict = {
            "ARROW_LENGTH":  self.ARROW_LENGTH,
            "ARROW_LIFETIME":  self.ARROW_LIFETIME,
            "BAC_SIZE": self.BAC_SIZE,
            "BACKGROUND_FACTOR": self.BACKGROUND_FACTOR,
        }

    def save(self, path):
        """
        Save the full TrackScoPy object as a pickle file.

        Parameters:
        - path: file path to save the object.
        """
        dbfile = open(path, 'wb')  # open file in binary mode
        pickle.dump(self, dbfile)  # serialize the object
        dbfile.close()  # close file

    def save_detection_results_only(self,path):
        """
        Save only the swim mode detection results (swimmer dataframes and parameters).

        Parameters:
        - path: output path to save results.
        """
        save_swim_mode_detection_results(self.swimmers,self.red_channel.parameter_dict,self.green_channel.parameter_dict,self.parameter_dict,path)

    def save_results_individual(self,path,filename,ORIENTATION=False):
        """
        Save detailed individual results for each swimmer, including:
        - trajectories
        - velocities
        - orientation/flow (optional)
        - episode-specific data
        - analysis parameters

        Parameters:
        - path: base directory to save results
        - filename: prefix for the result folder
        - ORIENTATION: if True, include orientation and flow information

        Workflow:
        - Create main result folder
        - Save global trajectory and velocity CSVs
        - For each swimmer:
            - Create individual folder
            - Save track and velocity CSVs
            - Compute swim episodes and save episode CSVs
            - If ORIENTATION=True, save orientation/flow CSVs and statistics
        - Save analysis parameters from red channel, green channel, and swim detection
        """

        # Define main results directory
        res_dir = os.path.join(path,f'{filename}_individual_analysis')

        # Create or overwrite results directory
        if os.path.exists(res_dir) == False:
            os.mkdir(res_dir)
        else:
            shutil.rmtree(res_dir)
            os.mkdir(res_dir)
            pass

        # Save global trajectory CSV
        coord_txt = os.path.join(res_dir,'trajectory_with_id.csv')
        self.data_with_id.to_csv(coord_txt, columns=['FRAME', 'SWIM-MODE', 'C:X-COORD', 'C:Y-COORD', 
                                                     'F:X-COORD', 'F:Y-COORD', 'CELL-ID','FLAGELLA-ID'], index=False)

        # Save global velocity CSV
        vel_txt = os.path.join(res_dir,'velocities_with_id.csv')
        self.data_with_id.to_csv(vel_txt, columns=['FRAME', 'SWIM-MODE', 'VELOCITY', 'CELL-ID','FLAGELLA-ID'], index=False)

        # Save orientation and flow CSVs if requested
        if ORIENTATION == True:
            of_names_txt = os.path.join(res_dir,'OF_names_with_id.csv')
            self.data_with_id.to_csv(of_names_txt, columns=['FRAME', 'SWIM-MODE', 'ORIENTATION', 'FLOW-DIRECTION', 'CELL-ID','FLAGELLA-ID'], index=False)

            of_values_txt = os.path.join(res_dir,'OF_values_with_id.csv')
            self.data_with_id.to_csv(of_values_txt, columns=['FRAME', 'SWIM-MODE', 'ORIENTATION-ANGLE', 'FLOW-ANGLE', 'CELL-ID','FLAGELLA-ID'], index=False)

        # Loop over each swimmer
        for i,swimmer in enumerate(self.swimmers):

            # Prepare dataframe with standardized column names
            df = swimmer.dataframe.rename(columns={'C:X-COORD': 'C:X', 'C:Y-COORD':'C:Y','F:X-COORD': 'F:X', 'F:Y-COORD':'F:Y',
                                                   'FLOW-DIRECTION':'FLOW','ORIENTATION-ANGLE': 'O-ANGLE','FLOW-ANGLE': 'F-ANGLE'})

            # Define individual swimmer folder
            swim_dir = os.path.join(res_dir,f'swimmer_{i+1}')

            # Create or overwrite swimmer folder
            if os.path.exists(swim_dir) == False:
                os.mkdir(swim_dir)
            else:
                shutil.rmtree(swim_dir)
                os.mkdir(swim_dir)
                pass

            # Save track CSV for swimmer
            coord_txt_in = os.path.join(swim_dir,'track.csv')
            df.to_csv(coord_txt_in,columns=['FRAME', 'SWIM-MODE', 'C:X', 'C:Y', 'F:X', 'F:Y'], index=False)

            # Save velocity CSV for swimmer
            swim_file = os.path.join(swim_dir,'velocity.csv')
            df.to_csv(swim_file,columns=['FRAME', 'SWIM-MODE', 'VELOCITY'], index=False)

            # Compute episode indices and velocity statistics
            modes = np.array(df['SWIM-MODE'],str)
            velocities = np.array(df['VELOCITY'],float)
            swimf = len(modes[~np.isnan(velocities)!=False])
            epi_indices = extract_episode_indices(modes)
            lines = []
            vel_stat_txt = os.path.join(swim_dir,'velocity_statistics.txt')

            # Overall swim statistics
            if swimf>1:
                lines.append('Run: %0.2f microns/sec for %0.0f frames'%(np.nanmean(velocities),swimf))

            # Loop over swim episodes
            for j in range(len(epi_indices)):
                episodes = modes[epi_indices[j]]
                epi_vels = velocities[epi_indices[j]]
                epif = len(episodes)

                if episodes[0] == 'push':
                    lines.append('Push: %0.2f microns/sec for %0.0f frames'%(np.nanmean(epi_vels),epif))
                elif episodes[0] == 'pull':
                    lines.append('Pull: %0.2f microns/sec for %0.0f frames'%(np.nanmean(epi_vels),epif))
                elif episodes[0] == 'wrap':
                    lines.append('Wrap: %0.2f microns/sec for %0.0f frames'%(np.nanmean(epi_vels),epif))

                # Save velocity statistics
                with open(vel_stat_txt, 'w') as f:
                    f.write('\n'.join(lines))

            # Orientation and flow processing if requested
            if ORIENTATION == True:

                # Save raw orientation/flow CSVs
                of_name_file = os.path.join(swim_dir,'OF_names.csv')
                df.to_csv(of_name_file,columns=['FRAME', 'SWIM-MODE', 'ORIENTATION', 'FLOW'], index=False)
                of_value_file = os.path.join(swim_dir,'OF_values.csv')
                df.to_csv(of_value_file,columns=['FRAME', 'SWIM-MODE', 'O-ANGLE', 'F-ANGLE'], index=False)

                # Define stats file paths
                or_stat_txt = os.path.join(swim_dir,'orientation_statistics.txt')
                fl_stat_txt = os.path.join(swim_dir,'flow_statistics.txt')

                # Extract orientation/flow arrays
                orientations = np.array(df['ORIENTATION'],str)
                flows = np.array(df['FLOW'],str)
                orientation_angles = np.array(df['O-ANGLE'],float)
                flow_angles = np.array(df['F-ANGLE'],float)

                # Orientation statistics
                orf = len(modes[~np.isnan(orientation_angles)!=False])
                lines2 = []
                if orf>1:
                    lines2.append('Orientation: %0.2f degrees for %0.0f frames'%(np.nanmean(orientation_angles),orf))

                epi_indices2 = extract_episode_indices(orientations)

                # Loop over orientation episodes
                for k in range(len(epi_indices2)):
                    episodes2 = orientations[epi_indices2[k]]
                    epi_ors = orientation_angles[epi_indices2[k]]
                    epif = len(episodes2)

                    if episodes2[0] == 'perpendicular':
                        lines2.append('Perpendicular: %0.2f degrees for %0.0f frames'%(np.nanmean(epi_ors),epif))
                    elif episodes2[0] == 'parallel':
                        lines2.append('Parallel: %0.2f degrees for %0.0f frames'%(np.nanmean(epi_ors),epif))

                    with open(or_stat_txt, 'w') as f:
                        f.write('\n'.join(lines2)) 

                # Flow statistics
                flf = len(modes[~np.isnan(flow_angles)!=False])
                lines3 = []
                if flf>1:
                    lines3.append('Flow: %0.2f degrees for %0.0f frames'%(np.nanmean(flow_angles),flf))

                epi_indices3 = extract_episode_indices(flows)

                # Loop over flow episodes
                for m in range(len(epi_indices3)):
                    episodes3 = flows[epi_indices3[m]]
                    epi_fls = flow_angles[epi_indices3[m]]
                    epif = len(episodes3)

                    if episodes3[0] == 'up':
                        lines3.append('Up: %0.2f degrees for %0.0f frames'%(np.nanmean(epi_fls),epif))
                    if episodes3[0] == 'down':
                        lines3.append('Down: %0.2f degrees for %0.0f frames'%(np.nanmean(epi_fls),epif))

                    with open(fl_stat_txt, 'w') as f:
                        f.write('\n'.join(lines3)) 

            # Define episodes directory for swimmer
            epi_dir = os.path.join(swim_dir,f'episodes_{swimmer.sorted_episodes_by}')

            # Create or overwrite episodes directory
            if os.path.exists(epi_dir) == False:
                os.mkdir(epi_dir)
            else:
                shutil.rmtree(epi_dir)
                os.mkdir(epi_dir)
                pass

            # Loop over episodes and save CSVs
            for j,episode in enumerate(swimmer.episodes):
                df_ep = episode.dataframe.rename(columns={'C:X-COORD': 'C:X', 'C:Y-COORD':'C:Y','F:X-COORD': 'F:X', 'F:Y-COORD':'F:Y',
                                                         'FLOW-DIRECTION':'FLOW','ORIENTATION-ANGLE': 'O-ANGLE','FLOW-ANGLE': 'F-ANGLE'})
                coord_txt_in = os.path.join(epi_dir,f'episode_{j+1}.csv')
                df_ep.to_csv(coord_txt_in,columns=['FRAME', 'SWIM-MODE', 'C:X', 'C:Y', 'F:X', 'F:Y'], index=False) 	

        # Save analysis parameters to text file
        param_path = os.path.join(res_dir,'parameters.txt')
        save_lines = []
        dicts = [self.red_channel.parameter_dict,self.green_channel.parameter_dict,self.parameter_dict]
        labels = ['image analysis of red channel', 'image analysis of green channel', 'swim_mode_detection']

        # Add timestamp
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_lines.append(f'This file was generated by TrackScoPy: main_image_analysis.ipynb on {current_datetime}')
        save_lines.append('')
        save_lines.append('')

        # Write parameters to file
        with open(param_path, 'w') as f:
            for i, d in enumerate(dicts):
                f.write('\n')
                f.write('======================================================')
                f.write('\n')
                f.write(f'Parameters for {labels[i]}:\n')
                f.write('\n')
                for k, v in d.items():
                    f.write(f'{k} = {v}\n')



# =====================
# Swimmer Helper Class
# =====================

class Swimmer():
    """
    Represents a single swimmer (cell/trajectory).
    Stores trajectory dataframe and extracts episodes of motion.
    """

    def __init__(self,df):
        """
        Initialize Swimmer object.

        Parameters:
        - df: pandas DataFrame with trajectory information.
              Must include columns 'VELOCITY', 'FRAME', 'SWIM-MODE'.
        """
        self.dataframe = df
        self.mean_speed = df['VELOCITY'].mean()   # average speed of swimmer
        self.run_time_frames = df['FRAME'].iloc[-1] - df['FRAME'].iloc[0]  # duration in frames
        self.extract_episodes()  # segment trajectory into episodes

    def extract_episodes(self,sort_by='SWIM-MODE'):
        """
        Segment trajectory into episodes based on column values.

        Parameters:
        - sort_by: column name (default 'SWIM-MODE') used to separate episodes.
        """
        epi_indices = extract_episode_indices(np.array(self.dataframe[sort_by]))  # episode index ranges
        self.sorted_episodes_by = sort_by
        # Create Episode objects for each segment
        self.episodes = [Episode(self.dataframe.iloc[x],self.sorted_episodes_by) for x in epi_indices]


# =====================
# Episode Helper Class
# =====================

class Episode():
    """
    Represents a continuous episode of a swimmer's motion,
    defined by consecutive rows in dataframe with the same property.
    """

    def __init__(self,df,sorted_by):
        """
        Initialize Episode object.

        Parameters:
        - df: pandas DataFrame slice belonging to this episode
        - sorted_by: column name used for episode segmentation
        """
        self.dataframe = df
        self.mean_speed = df['VELOCITY'].mean()   # mean speed in this episode
        self.run_time_frames = df['FRAME'].iloc[-1] - df['FRAME'].iloc[0]  # duration in frames
        self.sorted_by = sorted_by
        self.identity = df[self.sorted_by].iloc[0]  # label (e.g., swim mode) of episode

# ===========================
# Save Analysis Helper Class
# ===========================

class save_swim_mode_detection_results():
    """
    Save swim mode detection results, including parameters and swimmer objects.
    """

    def __init__(self,swimmers,red_channel_params,green_channel_params,parameter_dict,path):
        """
        Initialize and immediately save results.

        Parameters:
        - swimmers: list of Swimmer objects
        - red_channel_params: parameters for red channel processing
        - green_channel_params: parameters for green channel processing
        - parameter_dict: dictionary of swim-mode detection parameters
        - path: file path to save results
        """
        self.swimmers = swimmers
        self.path = path
        self.parameters_red_channel = red_channel_params
        self.parameters_green_channel = green_channel_params
        self.parameters_swim_mode_detection = parameter_dict
        self.save()

    def save(self):
        """
        Save results to file using pickle.
        """
        dbfile = open(self.path, 'wb')  # open file in binary mode
        pickle.dump(self, dbfile)       # dump entire object
        dbfile.close()                  # close file


# ===================
# Utility Functions
# ===================

def extract_episode_indices(X):
    """
    Identify start and end indices of episodes based on value changes.

    Parameters:
    - X: 1D array-like sequence (e.g., swim modes across frames)

    Returns:
    - Ep: list of numpy arrays with indices for each episode
    """
    ep_in = [0]
    for i in range(1,len(X)):
        if X[i]!=X[i-1]:   # detect change in value
            ep_in.append(i)
    ep_in.append(len(X))   # add last boundary

    Ep = []
    for j in range(len(ep_in)-1):
        Ep.append(np.arange(ep_in[j],ep_in[j+1]))  # collect ranges
        
    return Ep


def combine_channel(red_img,green_img,path=None):
    """
    Combine red and green image stacks into RGB images.

    Parameters:
    - red_img: list/array of grayscale images for red channel
    - green_img: list/array of grayscale images for green channel
    - path: optional path to save inverted combined image

    Returns:
    - seg_img_inv: RGB image stack with red/green combined and inverted
    """
    nImg = len(red_img)
    ht = len(red_img[0])
    wt = len(red_img[0][0])

    seg_img = np.zeros([nImg,ht,wt,3],np.uint8)  # original RGB
    b = np.zeros([ht,wt],np.uint8)               # empty blue channel
    
    # Build standard RGB image (R,G channels filled, B empty)
    for i in range(nImg):
        S = np.zeros([ht,wt,3],np.uint8)
        S[:,:,0] = red_img[i]
        S[:,:,1] = green_img[i]
        S[:,:,2] = b
        r,g,b = cv.split(S)
        seg_img[i] = cv.merge([r,g,b])
        
    seg_img_inv = np.zeros_like(seg_img)  # container for inverted result
    
    # Create inverted version with logical masks
    for i in range(nImg):
        r,g,b = cv.split(255-seg_img[i])
        g1 = np.zeros(g.shape,np.uint8)
        g1[:,:] = g[:,:]
        
        # Red mask
        g[r != 255] = 0 
        b[r != 255] = 0
        r[r != 255] = 255
        
        # Green mask
        r[g1 != 255] = 0
        b[g1 != 255] = 0
        g[g1 != 255] = 255
        
        seg_img_inv[i] = cv.merge([r,g,b])

    # Optionally save
    if path is None:
        pass
    else:
        cv.imwrite(path,seg_img_inv)

    return seg_img_inv


def generate_RGB_image(image,dtype=np.uint8,enhance=2):
    """
    Generate RGB version of grayscale image stack.

    Parameters:
    - image: input grayscale stack (3D array)
    - dtype: output data type (default uint8, can be uint16)
    - enhance: factor to enhance brightness

    Returns:
    - Img_rgb: 4D RGB image stack
    """
    nImg = len(image)
    ht = len(image[0])
    wt = len(image[0][0])
    input_dtype = image.dtype
    
    Img_rgb = np.zeros([nImg,ht,wt,3],dtype)

    # get bit depth and max value
    bit_depth = np.iinfo(input_dtype).bits
    maxi = float(2**bit_depth - 1)
        
    if dtype == np.uint8:
        fac = 255
    else:
        fac  = 2**16 - 1
    
    # Process each frame
    for i in tqdm(range(nImg),desc='Generating RGB channels'):
        r,g,b = cv.split(Img_rgb[i])
        img = ((fac*(image[i]/maxi))).astype(dtype)  # scale grayscale
        # copy grayscale into all three channels, with enhancement
        r[:,:] = 1.0*enhance*img[:,:]
        g[:,:] = 1.0*enhance*img[:,:]
        b[:,:] = 1.0*enhance*img[:,:]
        Img_rgb[i] = cv.merge([r,g,b])
        
    return Img_rgb 


def generate_calibration(tracklist):
    """
    Estimate drift velocity calibration from trajectories.

    Parameters:
    - tracklist: list of trajectories, each as array [x,y,t]

    Returns:
    - drift_velocity: array [[mean vx], [mean vy], [std vx], [std vy]]
    """
    dvx = []
    dvy = []
    
    for track in tracklist:    
        t = track[:,2]
        x = track[:,0]
        y = track[:,1]
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        
        dvx.extend(dx/dt)
        dvy.extend(dy/dt)
    
    drift_velocity = np.array([[np.mean(dvx)],[np.mean(dvy)],[np.std(dvx)],[np.std(dvy)]],float)
    return drift_velocity
    

def evaluate_angle(A,B):    
    """
    Compute signed angle between two 2D vectors.

    Parameters:
    - A: 2D vector (array-like length 2)
    - B: 2D vector (array-like length 2)

    Returns:
    - angle in radians between A and B (positive = counter-clockwise)
    """
    dot = np.dot(A,B)
    det = A[0]*B[1] - A[1]*B[0]  # determinant gives orientation
    return np.arctan2(det,dot)


