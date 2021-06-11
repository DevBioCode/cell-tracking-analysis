## START

def tracks_analysis(tracks_folder, spots, goodtracks, outline, image_outline,
                    dev_time, results_path, outline_region, animation_check):

    # for timing
    from datetime import datetime
    
    # to analyze TrackMate/Assisted Nuclei 3D tracking spreadsheets
    # as dataframes
    import pandas as pd
    
    # for various math operations
    import math
    
    # for file handling
    import os

    # also for file handling
    import shutil
    
    # for dealing with race conditions
    from tenacity import retry

    # for graphs
    import matplotlib.pyplot as plt
    
    # for generating outline shapes
    import matplotlib.path as path
    
    # for hsv to rgb conversion in graphs
    import matplotlib.colors
    
    # for various math operations
    import numpy as np
    
    # for calculating geometric and harmonic means
    from scipy.stats import gmean, hmean
    
    # for peak detection
    from scipy.signal import argrelextrema
    
    # for linear regression
    from sklearn.linear_model import LinearRegression
    
    # for scoring the regression
    from sklearn.metrics import r2_score
    
    # for animations
    from celluloid import Camera
    
    # to ignore warnings
    import warnings

    # time how long the whole script takes to run, but after user input
    script_start_time = datetime.now()

    # to append timestamps to filenames when exporting
    datetimestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")

    # turn off warnings about various math on zeros
    warnings.filterwarnings('ignore')

    specific_pairwise = ''
    # round start times up so tracks that start a bit later can be
    # included, and bin them to a multiple of 2 so embryos with varying
    # start times can be compared
    specific_pairwise_start = int(((dev_time // 2) + 1) * 2)
    # 6 hours is a reasonable length of time to observe patterns while
    # still incorporating many tracks
    specific_pairwise_end = specific_pairwise_start + 6

    # commented this out as detailed logging is almost never needed,
    # but can be reenabled by uncommenting
    # detailed_log = input("Enter 'y' for detailed logging: ")
    detailed_log = ''

    os.chdir(tracks_folder)

    work_dir = os.getcwd()
    print("Operating in", work_dir, "\n")
    print("Analyzing", outline_region, "\n")
    
    @retry
    def make_results_dir():
        if os.path.exists(f"AnalysisResults/{results_path}"):
            shutil.rmtree(f"AnalysisResults/{results_path}")        
        os.mkdir(f"AnalysisResults/{results_path}")
        os.mkdir(f"AnalysisResults/{results_path}/IndividualTracks")
    
    make_results_dir() 

    # return slope from start and end coordinates
    def slope(x1, y1, x2, y2):
        try:
            return (float(y2)-float(y1))/(float(x2)-float(x1))
        except:
            return 0

    # return angle between two lines of given slopes
    def angle(s1, s2):
        try:
            return math.degrees(math.atan((s2-s1)/(1 + (s2 * s1))))
        except:
            return 90

    # return distance between two points
    def dist2d(x1,y1,x2,y2):
        return (((float(x1) - float(x2)) ** 2) 
                + ((float(y1) - float(y2)) ** 2)) ** 0.5

    # subtract the specific pairwise start time from the original time
    def sub_start_time(orig_time):
        return abs((orig_time + (dev_time * 3600)) 
                   - (specific_pairwise_start * 3600))

    # subtract the specific pairwise end time from the original time
    def sub_end_time(orig_time):
        return abs((orig_time + (dev_time * 3600)) 
                   - (specific_pairwise_end * 3600))
    
    # create an arrow pointing at an angle, used to represent cells in 
    # animations (from https://stackoverflow.com/a/47858920/13630295)
    def get_arrow(ang):
        a = np.deg2rad(ang)
        # transpose to allow matrix multiplication of vertices with the
        # rotation matrix
        ar = np.array([[-.25,-.5],[.25,-.5],[0,.25],[-.25,-.5]]).T
        # rotate vertices using the 2x2 rotation matrix for a 2D 
        # polygon for angle a (in radians)
        rot = np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
        # transpose back for vertices as required by matplotlib
        return np.dot(rot,ar).T
    
    # sort tracks and within tracks by times
    spots_sorted = spots.sort_values(["TRACK_ID", "POSITION_T"])

    spots_sorted = spots_sorted.reset_index()
    
    spots_sorted_good = spots_sorted[0:0].drop(columns = ['index'])

    # determine the position of the first time point
    all_t_min_loc = spots_sorted["POSITION_T"].idxmin()

    # calculate time step by subtracting the first time point from the
    # second time point
    time_step = round(spots_sorted["POSITION_T"][all_t_min_loc + 1] 
                      - spots_sorted["POSITION_T"][all_t_min_loc])

    steps_15_min = int(round((15 * 60) / time_step))
     
    print(f'Time step is {time_step} seconds and there are {steps_15_min} \
          steps for every 15 minutes.')
    print("\n")

    ## CREATE A NEW DATAFRAME THAT HAS DETAILED TRACK STATISTICS

    # load a list of all unique track IDs as a numpy array(the general
    # format is df[column_name].unique()):
    tracks = spots_sorted["TRACK_ID"].unique()
    
    outline_list = []
    
    for i in range(len(outline)):
        outline_list.append([outline['X'][i], outline['Y'][i]])
    
    # convert outline x-y points to a path
    arch_points = path.Path(outline_list)
    
    # create an empty list that will be iteratively appended to as the
    # following for loop runs
    list_of_start_end_tracks = []

    list_of_start_end_spec_tracks = []

    list_of_rolling_stats = []
    
    list_of_distance_stats = []

    automatic_specific_tracker = 0
    
    time_counts = spots_sorted["POSITION_T"].value_counts(sort = False)
    
    time_counts = time_counts.sort_index()
    
    time_counts = time_counts.to_frame()
    
    time_counts = time_counts.reset_index()
        
    time_counts = time_counts.rename(columns={"index": "time", 
                                              "POSITION_T": "count"})
    
    min_tracks_for_average = 3
    
    min_tracks_times = []
    
    for i in range(len(time_counts) - 1):
        
        first_time_point = time_counts["count"][i]
        second_time_point = time_counts["count"][i+1]
        
        if (second_time_point >= min_tracks_for_average 
                and first_time_point < min_tracks_for_average):
            
            min_tracks_dict = {
                'min_tracks_start':int(time_counts["time"][i + 1])
                }
            min_tracks_times.append(min_tracks_dict)
            
        elif (first_time_point >= min_tracks_for_average 
                  and second_time_point < min_tracks_for_average):
            
            min_tracks_dict = {
                'min_tracks_end':int(time_counts["time"][i])
                }
            min_tracks_times.append(min_tracks_dict)
            
    for dict_entry in min_tracks_times:

        if 'min_tracks_start' in dict_entry:
            break
        else:

            min_tracks_dict = {
                'min_tracks_start':int(time_counts.iloc[0]["time"])
                }
            min_tracks_times.append(min_tracks_dict)
        
    for dict_entry in min_tracks_times:

        if 'min_tracks_end' in dict_entry:
            break
        else:

            min_tracks_dict = {
                'min_tracks_end':int(time_counts.iloc[-1]["time"])
                }
            min_tracks_times.append(min_tracks_dict)       
    
    if len(min_tracks_times) == 0:
        min_tracks_dict = {
            'min_tracks_start':int(time_counts.iloc[0]["time"]),
            'min_tracks_end':int(time_counts.iloc[-1]["time"])
            }
        min_tracks_times.append(min_tracks_dict)    
    
    min_tracks_times_df = pd.DataFrame(min_tracks_times)
    
    min_tracks_times_df = min_tracks_times_df.apply(lambda x: x.dropna().
                                                    reset_index(drop = True))
    min_tracks_times_df = min_tracks_times_df.fillna(time_counts["time"].
                                                     iloc[-1])

    min_tracks_times_df['min_tracks_run'] = min_tracks_times_df.apply(
        lambda x: x['min_tracks_end'] - x['min_tracks_start'], axis = 1)
    
    longest_run_index = min_tracks_times_df["min_tracks_run"].idxmax()
    
    longest_run_start = (min_tracks_times_df["min_tracks_start"]
                         [longest_run_index])
    longest_run_end = min_tracks_times_df["min_tracks_end"][longest_run_index]

    longest_run_start_hpf = round(((longest_run_start / 3600) + dev_time), 2)
    longest_run_end_hpf = round(((longest_run_end / 3600) + dev_time), 2)    
       

    for track in tracks:
        
        # create a copied dataframe spots_track that's spots with just
        # one TRACK_ID
        spots_track = spots_sorted.loc[spots_sorted["TRACK_ID"] 
                                       == track].copy()
        # finds the index location of the lowest time point in the 
        # track
        t_min_loc = spots_track["POSITION_T"].idxmin()
        # find the index location of the highest time point in the
        # track, then subtracts 1 to drop the last time point
        t_max_loc = (spots_track["POSITION_T"].idxmax()) - 1
        # start and end positions based on start and end times
        t_start_pos = spots_track["POSITION_T"][t_min_loc]
        t_end_pos = spots_track["POSITION_T"][t_max_loc]
        t_change = t_end_pos - t_start_pos
        x_start_pos = spots_track["POSITION_X"][t_min_loc]
        x_end_pos = spots_track["POSITION_X"][t_max_loc]
        y_start_pos = spots_track["POSITION_Y"][t_min_loc]
        y_end_pos = spots_track["POSITION_Y"][t_max_loc]
        # calculate 2D distance
        xy_change = dist2d(x_start_pos, y_start_pos, x_end_pos, y_end_pos)
        z_start_pos = spots_track["POSITION_Z"][t_min_loc]
        z_end_pos = spots_track["POSITION_Z"][t_max_loc]
        z_change = z_end_pos - z_start_pos
        # calculate velocity, not speed, as it's based on displacement,
        # microns per second
        xy_vel = xy_change / t_change
        
        xy_point = [x_end_pos, y_end_pos]
           
        spots_track.drop(columns 
                         = ['index']).to_csv(
                             f'AnalysisResults/{results_path}\/Individual\
                                 Tracks/spots_track_{track}.csv', 
                                 index = False, encoding = 'utf-8-sig')

        if (track in goodtracks[" Track ID "].tolist() 
                and arch_points.contains_point(xy_point)):
            
            spots_sorted_good = spots_sorted_good.append(
                spots_track.drop(columns = ['index']))
        
            new_start_end_track = {
                'track_num':track,
                't_start':t_start_pos,
                't_end':t_end_pos,
                't_delta':t_change,
                'x_start':x_start_pos,
                'x_end':x_end_pos,
                'y_start':y_start_pos,
                'y_end':y_end_pos,
                'xy_disp':xy_change,
                'xy_velocity':xy_vel,
                'z_start':z_start_pos,
                'z_end':z_end_pos,
                'z_delta':z_change
                } # creates a dictionary
            
            # make list_of_tracks a list of dictionaries
            list_of_start_end_tracks.append(new_start_end_track)

            # creating pairwise comparisons matched by time
            if ((t_start_pos + (dev_time * 3600)) 
                    <= (specific_pairwise_start * 3600) 
                    and (t_end_pos + (dev_time * 3600)) 
                    >= (specific_pairwise_end * 3600)):       
                
                automatic_specific_tracker += 1
                spots_track["sub_start"] = spots_track[
                    "POSITION_T"].apply(sub_start_time)
                spots_track["sub_end"] = spots_track[
                    "POSITION_T"].apply(sub_end_time)
                # find index location closest to specified start time
                t_spec_start_loc = spots_track["sub_start"].idxmin()
                # find index location closest to specified end time
                t_spec_end_loc = spots_track["sub_end"].idxmin()
                t_spec_start_pos = spots_track["POSITION_T"][t_spec_start_loc]
                t_spec_end_pos = spots_track["POSITION_T"][t_spec_end_loc]
                t_spec_change = t_spec_end_pos - t_spec_start_pos
                x_spec_start_pos = spots_track["POSITION_X"][t_spec_start_loc]
                x_spec_end_pos = spots_track["POSITION_X"][t_spec_end_loc]
                y_spec_start_pos = spots_track["POSITION_Y"][t_spec_start_loc]
                y_spec_end_pos = spots_track["POSITION_Y"][t_spec_end_loc]
                # calculate 2D distance
                xy_spec_change = dist2d(x_spec_start_pos, y_spec_start_pos, 
                                        x_spec_end_pos, y_spec_end_pos)
                z_spec_start_pos = spots_track["POSITION_Z"][t_spec_start_loc]
                z_spec_end_pos = spots_track["POSITION_Z"][t_spec_end_loc]
                z_spec_change = z_spec_end_pos - z_spec_start_pos
                xy_spec_vel = xy_spec_change / t_spec_change

                
                new_spec_start_end_track = {
                    'track_spec_num':track,
                    't_spec_start':t_spec_start_pos,
                    't_spec_end':t_spec_end_pos,
                    't_spec_delta':t_spec_change,
                    'x_spec_start':x_spec_start_pos,
                    'x_spec_end':x_spec_end_pos,
                    'y_spec_start':y_spec_start_pos,
                    'y_spec_end':y_spec_end_pos,
                    'xy_spec_disp':xy_spec_change,
                    'xy_spec_velocity':xy_spec_vel,
                    'z_spec_start':z_spec_start_pos,
                    'z_spec_end':z_spec_end_pos,
                    'z_spec_delta':z_spec_change
                    }
                
                
                list_of_start_end_spec_tracks.append(new_spec_start_end_track)
                
            # rolling measurements
            for i in range(t_min_loc, t_max_loc - (steps_15_min - 1)):
                rol_x_start = spots_track["POSITION_X"][i]
                rol_y_start = spots_track["POSITION_Y"][i]
                rol_x_end = spots_track["POSITION_X"][i + steps_15_min]
                rol_y_end = spots_track["POSITION_Y"][i + steps_15_min]
                rol_xy_change = dist2d(rol_x_start, rol_y_start, 
                                       rol_x_end, rol_y_end)
                # convert velocity to microns per hour
                rol_xy_vel = (rol_xy_change * 3600) / (time_step 
                                                       * steps_15_min)
                rol_z_start = spots_track["POSITION_Z"][i]
                rol_z_end = spots_track["POSITION_Z"][i + steps_15_min]
                rol_z_change = rol_z_end - rol_z_start
                # convert velocity to microns per hour
                rol_z_vel = (rol_z_change * 3600) / (time_step * steps_15_min)
                rol_dist_accum = 0
                # calculate cumulative distance to 
                # calculate persistence
                for j in range(i, i + steps_15_min):
                    rol_dist_x_start = spots_track["POSITION_X"][j]
                    rol_dist_y_start = spots_track["POSITION_Y"][j]
                    rol_dist_x_end = spots_track["POSITION_X"][j + 1]
                    rol_dist_y_end = spots_track["POSITION_Y"][j + 1]
                    rol_dist_change = dist2d(
                        rol_dist_x_start, rol_dist_y_start, 
                        rol_dist_x_end, rol_dist_y_end)
                    rol_dist_speed = rol_dist_change * 3600 / time_step
                    rol_dist_accum += rol_dist_change
                    
                    new_distance_stats = {
                        'track':track,
                        'dev_time_hpf':round((spots_track["POSITION_T"][j + 1] 
                                              / 3600) + dev_time, 4),
                        'dist_change':rol_dist_change,
                        'xy_dist_speed':rol_dist_speed,
                        'dist_from_origin_squared':dist2d(
                            x_start_pos, y_start_pos, 
                            rol_dist_x_end, rol_dist_y_end) ** 2
                        }
                    
                    list_of_distance_stats.append(new_distance_stats)
                    
                try:
                    rol_pers = float(rol_xy_change / rol_dist_accum)
                except:
                    rol_pers = 0
                
                new_rolling_stats = {
                    'track':track,
                    'dev_time_hpf':round(((spots_track
                                           ["POSITION_T"][i + steps_15_min] 
                                           / 3600) + dev_time), 2),
                    'x_position(μm)':round(rol_x_end, 2),
                    'y_position(μm)':round(rol_y_end, 2),
                    'xy_speed(μm/h)':round(rol_xy_vel, 2),
                    'xy_angle(deg)':round(
                        angle(slope(rol_x_start, rol_y_start, 
                                    rol_x_end, rol_y_end), 0), 2),
                    'xy_speed_log':round(np.log(rol_xy_vel), 2),
                    'z_speed(μm/h)':round(rol_z_vel, 2),
                    'persistence':round(rol_pers, 2)
                    }
                
               
                list_of_rolling_stats.append(new_rolling_stats)
                
                print("\rTrack", track, " ", end = '')
            
    print("\n")
    
    #create a dataframe from list_of_tracks
    track_anal = pd.DataFrame(list_of_start_end_tracks)

    track_anal_spec = pd.DataFrame(list_of_start_end_spec_tracks)

    spots_rolling = pd.DataFrame(list_of_rolling_stats)
    
    distance_stats = pd.DataFrame(list_of_distance_stats) 
    
    
    # set plot size to 1800 x 1200 px
    plt.figure('med_lat', figsize = (18,12))
    
    for i in track_anal_spec.index:

        plt.scatter(track_anal_spec["x_spec_start"].iloc[i], 
                    - (track_anal_spec["y_spec_start"].iloc[i]), 
                    facecolors = 'none', edgecolors 
                    = matplotlib.colors.hsv_to_rgb(np.atleast_2d(
                        (0, ((track_anal_spec["z_spec_end"].iloc[i] 
                              - min(track_anal_spec["z_spec_end"])) 
                             / (max(track_anal_spec["z_spec_end"]) 
                                - min(track_anal_spec["z_spec_end"]))), 
                         ((track_anal_spec["z_spec_end"].iloc[i] 
                           - min(track_anal_spec["z_spec_end"])) 
                          / (max(track_anal_spec["z_spec_end"]) 
                             - min(track_anal_spec["z_spec_end"])))))))
        
        plt.scatter(track_anal_spec["x_spec_end"].iloc[i], 
                    - (track_anal_spec["y_spec_end"].iloc[i]), c 
                    = matplotlib.colors.hsv_to_rgb(np.atleast_2d(
                        (0, ((track_anal_spec["z_spec_end"].iloc[i] 
                              - min(track_anal_spec["z_spec_end"])) 
                             / (max(track_anal_spec["z_spec_end"]) 
                                - min(track_anal_spec["z_spec_end"]))), 
                         ((track_anal_spec["z_spec_end"].iloc[i] 
                           - min(track_anal_spec["z_spec_end"])) 
                          / (max(track_anal_spec["z_spec_end"]) 
                             - min(track_anal_spec["z_spec_end"])))))))
        
        plt.text(track_anal_spec["x_spec_start"].iloc[i], 
                 - (track_anal_spec["y_spec_start"].iloc[i]) + 2, 
                 track_anal_spec["track_spec_num"].iloc[i], 
                 ha = 'center', c = 'dimgray')
        plt.text(track_anal_spec["x_spec_end"].iloc[i], 
                 - (track_anal_spec["y_spec_end"].iloc[i]) + 2, 
                 track_anal_spec["track_spec_num"].iloc[i], 
                 ha = 'center', c = 'k')
        
        plt.xlim(0, image_outline["X"][2])
        plt.ylim(-image_outline["Y"][2],0)

        plt.xlabel("x distance in μm")
        plt.ylabel("y distance in μm")       
   
    plt.title(f'x-y positions shown at {specific_pairwise_start} hpf, \
              color based on z position at {specific_pairwise_end} hpf', 
              position = (0.5, 0.97))
    lat_rect = plt.Rectangle(((((image_outline["X"][2]) / 2) - 15), 
                              (-image_outline["Y"][2])), 30, 10, ec = 'k', fc 
                             = 'none')
    plt.gca().add_patch(lat_rect)
    plt.scatter((((image_outline["X"][2]) / 2) - 10), 
                ((-image_outline["Y"][2]) + 6), c = 'r')
    plt.scatter((((image_outline["X"][2]) / 2) + 10), 
                ((-image_outline["Y"][2]) + 6), c = 'k')
    plt.text((((image_outline["X"][2]) / 2) - 10), 
             ((-image_outline["Y"][2]) + 1), "lateral", ha = 'center')
    plt.text((((image_outline["X"][2]) / 2) + 10), 
             ((-image_outline["Y"][2]) + 1), "medial", ha = 'center')
    plt.savefig(f"AnalysisResults/{results_path}/\
                medial_lateral_correlation{datetimestamp}.png")

    plt.close()
        
    
    if not spots_rolling.empty:
        
        plt.figure(1, figsize=(18,12))
    
        fig1_counter = 0

        for track in tracks:
            
            if track in spots_rolling["track"].tolist():

                spots_rolling_track = spots_rolling.loc[spots_rolling["track"] 
                                                        == track]
        
                plot_1_x = spots_rolling_track['dev_time_hpf']
                plot_1_y = spots_rolling_track['xy_speed(μm/h)']
                        
                plt.plot(plot_1_x, plot_1_y, label = track)
                plt.xlim(10, 40)
                plt.ylim(0, 100)
                plt.legend(ncol = 2)
                
                fig1_counter += 1
            
            if fig1_counter == 10:
                break

        plt.xlabel("hours post fertilization")
        plt.ylabel("xy speed in μm/h, rolling 15 min average")
        plt.savefig(f"AnalysisResults/{results_path}/\
                    speeds_xy{datetimestamp}.png")
        plt.close(1)
        
        #animated
        
        if animation_check == 'y':
        
            alltracksanim = plt.figure(figsize=(18,12))
        
            cam = Camera(alltracksanim)
            
            for time in spots_rolling['dev_time_hpf'].unique():
        
                x_pos = spots_rolling['x_position(μm)'].loc[
                    spots_rolling['dev_time_hpf'] == time]
                y_pos = -(spots_rolling['y_position(μm)'].loc[
                    spots_rolling['dev_time_hpf'] == time])
                curr_track = spots_rolling['track'].loc[
                    spots_rolling['dev_time_hpf'] == time]
                curr_vel = spots_rolling['xy_speed(μm/h)'].loc[
                    spots_rolling['dev_time_hpf'] == time]
                curr_ang = spots_rolling['xy_angle(deg)'].loc[
                    spots_rolling['dev_time_hpf'] == time]

                for i in range(len(x_pos)):
                    
                    # graph cells with size based on velocity and
                    # color based on track number
                    # commented out but kept in case it's needed
                    # sc = plt.scatter(
                    #     x_pos.iloc[i], y_pos.iloc[i], s = 100 ** 
                    #     (1 + 0.5 * ((curr_vel.iloc[i] - min(
                    #         spots_rolling['xy_speed(μm/h)'])) 
                    #         / ((max(spots_rolling['xy_speed(μm/h)']) 
                    #             - min(spots_rolling['xy_speed(μm/h)']))))), 
                    #     c = matplotlib.colors.hsv_to_rgb(np.atleast_2d(
                    #         (curr_track.iloc[i] / max(tracks), 1.0, 1.0))), 
                    #     marker = get_arrow(curr_ang.iloc[i] - 180))
                    
                    # graph cells with constant size and color based
                    # on velocity
                    sc = plt.scatter(
                        x_pos.iloc[i], y_pos.iloc[i], s = 500, 
                        c = matplotlib.colors.hsv_to_rgb(np.atleast_2d(
                            (0.33, (curr_vel.iloc[i] 
                                    - min(spots_rolling['xy_speed(μm/h)'])) 
                             / ((max(spots_rolling['xy_speed(μm/h)']) 
                                 - min(spots_rolling['xy_speed(μm/h)']))), 
                             (curr_vel.iloc[i] 
                              - min(spots_rolling['xy_speed(μm/h)'])) 
                             / ((max(spots_rolling['xy_speed(μm/h)']) 
                                 - min(spots_rolling['xy_speed(μm/h)'])))))), 
                        marker = get_arrow(curr_ang.iloc[i] - 180))
                    
                    plt.legend([sc], [f'{time:.2f} hpf'], markerscale = 0, 
                               frameon = 0)
                    plt.text(x_pos.iloc[i], y_pos.iloc[i] + 5, 
                             (curr_track.iloc[i]), ha = 'center', 
                             c = 'dimgray')
                    plt.xlim(0, image_outline["X"][2])
                    plt.ylim(-image_outline["Y"][2],0)
                    plt.xlabel("x distance in μm")
                    plt.ylabel("y distance in μm")
        
                cam.snap()
        
            alltracksanimation = cam.animate()
        
            alltracksanimation.save(f'AnalysisResults/{results_path}/\
                                    animation{datetimestamp}.mp4', 
                                    fps = steps_15_min)
        
        
    
    
        for track in tracks:
            
            if track in spots_rolling["track"].tolist():
                
                # create a dataframe spots_track that's spots with
                # just one TRACK_ID
                spots_rolling_track = spots_rolling.loc[spots_rolling[
                    'track'] == track]
                distance_stats_track = distance_stats.loc[distance_stats[
                    'track'] == track]
        
                plt.figure('ind1', figsize=(18,12))
                
                plot_ind1_x = spots_rolling_track['dev_time_hpf']
                plot_ind1_y = spots_rolling_track['xy_speed(μm/h)']
                
                plot_ind1_x_dist = distance_stats_track['dev_time_hpf']
                plot_ind1_y_dist = distance_stats_track['xy_dist_speed']
                        
                plt.plot(plot_ind1_x, plot_ind1_y, label = track)
                plt.plot(plot_ind1_x_dist, plot_ind1_y_dist, linewidth = 0.1, 
                         alpha = 0.1)
               
                plt.xlim(10, 40)
                plt.ylim(0, 200)
                plt.legend(ncol = 2)
    
                plt.xlabel("hours post fertilization")
                plt.ylabel("xy speed in μm/h, rolling 15 min average")
                plt.savefig(f"AnalysisResults/{results_path}/\
                            IndividualTracks/speeds_xy_{track}.png")
                plt.close('ind1')
    
    
        plt.figure(2, figsize=(18,12))
    
        fig2_counter = 0
    
        for track in tracks:
            
            if track in spots_rolling["track"].tolist():
                
                spots_rolling_track = spots_rolling.loc[spots_rolling["track"] 
                                                        == track]
        
                plot_2_x = spots_rolling_track['dev_time_hpf']
                plot_2_y = spots_rolling_track['z_speed(μm/h)']
                        
                plt.plot(plot_2_x, plot_2_y, label = track)
                plt.xlim(10, 40)
                plt.ylim(-30, 60)
                plt.legend(ncol = 2)
                
                fig2_counter += 1
            
            if fig2_counter == 10:
                break
                
        plt.xlabel("hours post fertilization")
        plt.ylabel("z speed in μm/h, rolling 15 min average")
        plt.savefig(f"AnalysisResults/{results_path}/\
                    speeds_z{datetimestamp}.png")
        plt.close(2)
        
        for track in tracks:
            
            if track in spots_rolling["track"].tolist():
                
                spots_rolling_track = spots_rolling.loc[spots_rolling[
                    "track"] == track]
        
                plt.figure('ind2', figsize=(18,12))
                
                plot_ind2_x = spots_rolling_track['dev_time_hpf']
                plot_ind2_y = spots_rolling_track['z_speed(μm/h)']
                        
                plt.plot(plot_ind2_x, plot_ind2_y, label = track)
                plt.xlim(10, 40)
                plt.ylim(-30, 60)
                plt.legend(ncol = 2)
    
                plt.xlabel("hours post fertilization")
                plt.ylabel("z speed in μm/h, rolling 15 min average")
                plt.savefig(f"AnalysisResults/{results_path}/\
                            IndividualTracks/speeds_z_{track}.png")
                plt.close('ind2')
    
        plt.figure(3, figsize=(18,12))
    
        fig3_counter = 0
    
        for track in tracks:
            
            if track in spots_rolling["track"].tolist():
                
                spots_rolling_track = spots_rolling.loc[spots_rolling[
                    "track"] == track]
        
                plot_3_x = spots_rolling_track['dev_time_hpf']
                plot_3_y = spots_rolling_track['persistence']
                        
                plt.plot(plot_3_x, plot_3_y, label = track)
                plt.xlim(10, 40)
                plt.ylim(0, 1)
                plt.legend(ncol = 2)
                
                fig3_counter += 1
            
            if fig3_counter == 10:
                break
                
        plt.xlabel("hours post fertilization")
        plt.ylabel("persistence, rolling 15 min average")
        plt.savefig(f"AnalysisResults/{results_path}/\
                    persistence{datetimestamp}.png")
        plt.close(3)
    
        for track in tracks:
            
            if track in spots_rolling["track"].tolist():
                
                spots_rolling_track = spots_rolling.loc[spots_rolling[
                    "track"] == track]
        
                plt.figure('ind3', figsize=(18,12))
                
                plot_ind3_x = spots_rolling_track['dev_time_hpf']
                plot_ind3_y = spots_rolling_track['persistence']
                        
                plt.plot(plot_ind3_x, plot_ind3_y, label = track)
                plt.xlim(10, 40)
                plt.ylim(0, 1)
                plt.legend(ncol = 2)
    
                plt.xlabel("hours post fertilization")
                plt.ylabel("persistence, rolling 15 min average")
                plt.savefig(f"AnalysisResults/{results_path}/\
                            IndividualTracks/persistence_{track}.png")
                plt.close('ind3')
    
        list_of_mean_stats = []
    
        for common_time in spots_rolling['dev_time_hpf'].unique():
            dev_time_hpf = common_time
            num_common_tracks = (
                spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                  == common_time]['track']).count()
            common_time_xy_speed_mean = (
                spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                  == common_time]['xy_speed(μm/h)']).mean()
            common_time_xy_speed_mean_log = (
                spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                  == common_time]['xy_speed_log']).mean()
            common_time_xy_speed_gmean = gmean(
                spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                  == common_time]['xy_speed(μm/h)'])
            common_time_xy_speed_gmean_log = gmean(
                spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                  == common_time]['xy_speed_log'])
            common_time_xy_speed_hmean = hmean(
                spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                  == common_time]['xy_speed(μm/h)'])
            common_time_z_speed_mean = (
                spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                  == common_time]['z_speed(μm/h)']).mean()
            common_time_z_speed_gmean = gmean(
                spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                  == common_time]['z_speed(μm/h)'])
            common_time_z_speed_hmean = hmean(
                abs(spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                      == common_time]['z_speed(μm/h)']))
            common_time_pers_mean = (
                spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                  == common_time]['persistence']).mean()
            common_time_pers_gmean = gmean(
                spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                  == common_time]['persistence'])
            common_time_pers_hmean = hmean(
                spots_rolling.loc[spots_rolling["dev_time_hpf"] 
                                  == common_time]['persistence'])
                  
            common_time_mean_stats = {
                'common_time_hpf':dev_time_hpf,
                'num_tracks':num_common_tracks,
                'xy_speed_mean':common_time_xy_speed_mean,
                'xy_speed_mean_log':common_time_xy_speed_mean_log,
                'xy_speed_gmean':common_time_xy_speed_gmean,
                'xy_speed_gmean_log':common_time_xy_speed_gmean_log,
                'xy_speed_hmean':common_time_xy_speed_hmean,
                'z_speed_mean':common_time_z_speed_mean,
                'z_speed_gmean':common_time_z_speed_gmean,
                'z_speed_hmean':common_time_z_speed_hmean,
                'persistence_mean':common_time_pers_mean,
                'persistence_gmean':common_time_pers_gmean,
                'persistence_hmean':common_time_pers_hmean
                }
            
            list_of_mean_stats.append(common_time_mean_stats)
            
        time_mean_stats = pd.DataFrame(list_of_mean_stats)
        
        time_mean_stats = time_mean_stats.loc[
            (time_mean_stats['common_time_hpf'] 
             >= longest_run_start_hpf) & (time_mean_stats['common_time_hpf'] 
                                          <= longest_run_end_hpf)]
        
        max_tracks_in_longest_run = time_mean_stats['num_tracks'].max()
        
        if min_tracks_for_average == max_tracks_in_longest_run:
            mean_tracks = str(min_tracks_for_average)
        else:
            mean_tracks = (
                f'{min(min_tracks_for_average, max_tracks_in_longest_run)}'
                f' to \
                    {max(min_tracks_for_average, max_tracks_in_longest_run)}')
     
        list_of_dist_stats = []
        
        for dist_common_time in distance_stats['dev_time_hpf'].unique():
            dist_dev_time_hpf = dist_common_time
            dist_common_time_xy_speed_mean = (
                distance_stats.
                loc[distance_stats['dev_time_hpf'] 
                    == dist_common_time]['xy_dist_speed']).mean()
            dist_common_time_xy_speed_gmean = gmean(
                distance_stats.
                loc[distance_stats['dev_time_hpf'] 
                    == dist_common_time]['xy_dist_speed'])
            mean_squared_displacement = (
                distance_stats.
                loc[distance_stats['dev_time_hpf'] 
                    == dist_common_time]['dist_from_origin_squared']).mean()
            
            dist_common_time_mean_stats = {
                'common_time_hpf':dist_dev_time_hpf,
                'dist_xy_speed_mean':dist_common_time_xy_speed_mean,
                'dist_xy_speed_gmean':dist_common_time_xy_speed_gmean,
                'msd':mean_squared_displacement
                }
            
            list_of_dist_stats.append(dist_common_time_mean_stats)
            
        dist_mean_stats = pd.DataFrame(list_of_dist_stats)
    
        # 1.5 gives the best results when compared to 
        # multiplying by 1.0 or 2.0 when visually examined
        extrema_n = int(steps_15_min * 1.5)
            
        time_mean_stats[
            'xy_speed_min'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['xy_speed_mean'].values, 
                np.less_equal, order = extrema_n)[0]]['xy_speed_mean']
        time_mean_stats[
            'xy_speed_max'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['xy_speed_mean'].values, 
                np.greater_equal, order = extrema_n)[0]]['xy_speed_mean']
        time_mean_stats[
            'xy_speed_min_log'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['xy_speed_mean_log'].values, 
                np.less_equal, order = extrema_n)[0]]['xy_speed_mean_log']
        time_mean_stats[
            'xy_speed_max_log'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['xy_speed_mean_log'].values, 
                np.greater_equal, order = extrema_n)[0]]['xy_speed_mean_log']
        time_mean_stats[
            'xy_speed_min_gmean'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['xy_speed_gmean'].values, 
                np.less_equal, order = extrema_n)[0]]['xy_speed_gmean']
        time_mean_stats[
            'xy_speed_max_gmean'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['xy_speed_gmean'].values, 
                np.greater_equal, order = extrema_n)[0]]['xy_speed_gmean']
        time_mean_stats[
            'xy_speed_min_gmean_log'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['xy_speed_gmean_log'].values, 
                np.less_equal, order = extrema_n)[0]]['xy_speed_gmean_log']
        time_mean_stats[
            'xy_speed_max_gmean_log'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['xy_speed_gmean_log'].values, 
                np.greater_equal, order = extrema_n)[0]]['xy_speed_gmean_log']
        time_mean_stats[
            'z_speed_min'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['z_speed_mean'].values, 
                np.less_equal, order = extrema_n)[0]]['z_speed_mean']
        time_mean_stats[
            'z_speed_max'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['z_speed_mean'].values, 
                np.greater_equal, order = extrema_n)[0]]['z_speed_mean']
        time_mean_stats[
            'persistence_min'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['persistence_mean'].values, 
                np.less_equal, order = extrema_n)[0]]['persistence_mean']
        time_mean_stats[
            'persistence_max'] = time_mean_stats.iloc[argrelextrema(
                time_mean_stats['persistence_mean'].values, 
                np.greater_equal, order = extrema_n)[0]]['persistence_mean']
        
        mean_xy_mins = time_mean_stats[[
            'common_time_hpf', 'xy_speed_min']].copy()
        mean_xy_mins = mean_xy_mins.dropna()
        mean_xy_maxs = time_mean_stats[
            ['common_time_hpf', 'xy_speed_max']].copy()
        mean_xy_maxs = mean_xy_maxs.dropna()
        mean_xy_mins_log = time_mean_stats[
            ['common_time_hpf', 'xy_speed_min_log']].copy()
        mean_xy_mins_log = mean_xy_mins_log.dropna()
        mean_xy_maxs_log = time_mean_stats[
            ['common_time_hpf', 'xy_speed_max_log']].copy()
        mean_xy_maxs_log = mean_xy_maxs_log.dropna()
        mean_xy_mins_gmean = time_mean_stats[
            ['common_time_hpf', 'xy_speed_min_gmean']].copy()
        mean_xy_mins_gmean = mean_xy_mins_gmean.dropna()
        mean_xy_maxs_gmean = time_mean_stats[
            ['common_time_hpf', 'xy_speed_max_gmean']].copy()
        mean_xy_maxs_gmean = mean_xy_maxs_gmean.dropna()
        mean_xy_mins_gmean_log = time_mean_stats[
            ['common_time_hpf', 'xy_speed_min_gmean_log']].copy()
        mean_xy_mins_gmean_log = mean_xy_mins_gmean_log.dropna()
        mean_xy_maxs_gmean_log = time_mean_stats[
            ['common_time_hpf', 'xy_speed_max_gmean_log']].copy()
        mean_xy_maxs_gmean_log = mean_xy_maxs_gmean_log.dropna()
        
        mean_z_mins = time_mean_stats[
            ['common_time_hpf', 'z_speed_min']].copy()
        mean_z_mins = mean_z_mins.dropna()
        mean_z_maxs = time_mean_stats[
            ['common_time_hpf', 'z_speed_max']].copy()
        mean_z_maxs = mean_z_maxs.dropna()
        mean_persistence_mins = time_mean_stats[
            ['common_time_hpf', 'persistence_min']].copy()
        mean_persistence_mins = mean_persistence_mins.dropna()
        mean_persistence_maxs = time_mean_stats[
            ['common_time_hpf', 'persistence_max']].copy()
        mean_persistence_maxs = mean_persistence_maxs.dropna()
        
        mean_xy_mins['period_deltas'] = mean_xy_mins[
            'common_time_hpf'].diff()
        mean_xy_maxs['period_deltas'] = mean_xy_maxs[
            'common_time_hpf'].diff()
        mean_xy_mins_log['period_deltas'] = mean_xy_mins_log[
            'common_time_hpf'].diff()
        mean_xy_maxs_log['period_deltas'] = mean_xy_maxs_log[
            'common_time_hpf'].diff()
        mean_xy_mins_gmean['period_deltas'] = mean_xy_mins_gmean[
            'common_time_hpf'].diff()
        mean_xy_maxs_gmean['period_deltas'] = mean_xy_maxs_gmean[
            'common_time_hpf'].diff()
        mean_xy_mins_gmean_log['period_deltas'] = mean_xy_mins_gmean_log[
            'common_time_hpf'].diff()
        mean_xy_maxs_gmean_log['period_deltas'] = mean_xy_maxs_gmean_log[
            'common_time_hpf'].diff()
        
        mean_z_mins['period_deltas'] = mean_z_mins[
            'common_time_hpf'].diff()
        mean_z_maxs['period_deltas'] = mean_z_maxs[
            'common_time_hpf'].diff()
        mean_persistence_mins['period_deltas'] = mean_persistence_mins[
            'common_time_hpf'].diff()
        mean_persistence_maxs['period_deltas'] = mean_persistence_maxs[
            'common_time_hpf'].diff()  
    
        
        plt.figure(4, figsize=(18,12))
    
        plot_4_x = time_mean_stats['common_time_hpf']
        plot_4_y = time_mean_stats['xy_speed_mean']
        plot_4_x_dist = dist_mean_stats['common_time_hpf']
        plot_4_y_dist = dist_mean_stats['dist_xy_speed_mean']
    
        plt.plot(plot_4_x, plot_4_y)
        plt.plot(plot_4_x_dist, plot_4_y_dist, linewidth = 0.1, alpha = 0.8)
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['xy_speed_min'], color='m')
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['xy_speed_max'], color='c')
        plt.xlim(10, 40)
        plt.ylim(0, 200)
        plt.xlabel("hours post fertilization")
        plt.ylabel("mean xy speed in μm/h, rolling 15 min average")
        plt.title(f'Mean over {mean_tracks} tracks', position = (0.5, 0.97))
        plt.figtext(
            0.7596, 0.85, 
            f'Minima period: {round(mean_xy_mins["period_deltas"].mean(), 1)}' 
            f' hpf, s.d. {round(mean_xy_mins["period_deltas"].std(), 2)}')
        plt.figtext(
            0.757, 0.83, 
            f'Maxima period: {round(mean_xy_maxs["period_deltas"].mean(), 1)}'
            f' hpf, s.d. {round(mean_xy_maxs["period_deltas"].std(), 2)}')
        plt.savefig(
            f"AnalysisResults/{results_path}/\
                mean_xy_speed{datetimestamp}.png")
        plt.close(4)
        
        plt.figure('4_log', figsize=(18,12))
    
        plot_4_x = time_mean_stats['common_time_hpf']
        plot_4_y = time_mean_stats['xy_speed_mean_log']
    
        plt.plot(plot_4_x, plot_4_y)
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['xy_speed_min_log'], color='m')
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['xy_speed_max_log'], color='c')
        plt.xlim(10, 40)
        plt.ylim(0, 5)
        plt.xlabel("hours post fertilization")
        plt.ylabel("mean natural log xy speed in μm/h, \
                   rolling 15 min average")
        plt.title(f'Mean over {mean_tracks} tracks', position = (0.5, 0.97))
        plt.figtext(
            0.7596, 0.85, 
            f'Minima period: \
                {round(mean_xy_mins_log["period_deltas"].mean(), 1)} hpf, \
                    s.d. {round(mean_xy_mins_log["period_deltas"].std(), 2)}')
        plt.figtext(
            0.757, 0.83, 
            f'Maxima period: \
                {round(mean_xy_maxs_log["period_deltas"].mean(), 1)} hpf, \
                    s.d. {round(mean_xy_maxs_log["period_deltas"].std(), 2)}')
        plt.savefig(f"AnalysisResults/{results_path}/\
                    mean_log_xy_speed{datetimestamp}.png")
        plt.close('4_log')
        
        plt.figure('4_gmean', figsize=(18,12))
    
        plot_4_x = time_mean_stats['common_time_hpf']
        plot_4_y = time_mean_stats['xy_speed_gmean']
        plot_4_x_dist = dist_mean_stats['common_time_hpf']
        plot_4_y_dist = dist_mean_stats['dist_xy_speed_gmean']
    
        plt.plot(plot_4_x, plot_4_y)
        plt.plot(plot_4_x_dist, plot_4_y_dist, linewidth = 0.1, alpha = 0.8)
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['xy_speed_min_gmean'], color='m')
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['xy_speed_max_gmean'], color='c')
        plt.xlim(10, 40)
        plt.ylim(0, 200)
        plt.xlabel("hours post fertilization")
        plt.ylabel("geometric mean xy speed in μm/h, rolling 15 min average")
        plt.title(f'Mean over {mean_tracks} tracks', position = (0.5, 0.97))
        plt.figtext(
            0.7596, 0.85, 
            f'Minima period: \
                {round(mean_xy_mins_gmean["period_deltas"].mean(), 1)} \
                    hpf, s.d. \
                        {round(mean_xy_mins_gmean["period_deltas"].std(), 2)}'
                        )
        plt.figtext(
            0.757, 0.83, 
            f'Maxima period: \
                {round(mean_xy_maxs_gmean["period_deltas"].mean(), 1)} \
                    hpf, s.d. \
                        {round(mean_xy_maxs_gmean["period_deltas"].std(), 2)}'
                        )
        plt.savefig(f"AnalysisResults/{results_path}/\
                    gmean_xy_speed{datetimestamp}.png")
        plt.close('4_gmean')
        
        plt.figure('4_gmean_log', figsize=(18,12))
    
        plot_4_x = time_mean_stats['common_time_hpf']
        plot_4_y = time_mean_stats['xy_speed_gmean_log']
    
        plt.plot(plot_4_x, plot_4_y)
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['xy_speed_min_gmean_log'], color='m')
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['xy_speed_max_gmean_log'], color='c')
        plt.xlim(10, 40)
        plt.ylim(0, 5)
        plt.xlabel("hours post fertilization")
        plt.ylabel("geometric mean natural log xy speed in μm/h, \
                   rolling 15 min average")
        plt.title(f'Mean over {mean_tracks} tracks', position = (0.5, 0.97))
        plt.figtext(
            0.7596, 0.85, 
            f'Minima period: \
                {round(mean_xy_mins_gmean_log["period_deltas"].mean(), 1)}'
            f' hpf, s.d. \
                {round(mean_xy_mins_gmean_log["period_deltas"].std(), 2)}')
        plt.figtext(
            0.757, 0.83, 
            f'Maxima period: \
                {round(mean_xy_maxs_gmean_log["period_deltas"].mean(), 1)}' 
            f' hpf, s.d. \
                {round(mean_xy_maxs_gmean_log["period_deltas"].std(), 2)}')
        plt.savefig(f"AnalysisResults/{results_path}/\
                    gmean_log_xy_speed{datetimestamp}.png")
        plt.close('4_gmean_log')
        
        plt.figure('4ext', figsize=(18,12))
        
        plot_4min_x = mean_xy_mins['common_time_hpf']
        plot_4min_y = mean_xy_mins['period_deltas']
        plot_4max_x = mean_xy_maxs['common_time_hpf']
        plot_4max_y = mean_xy_maxs['period_deltas']
        
        plt.plot(plot_4min_x, plot_4min_y, color='m')
        plt.plot(plot_4max_x, plot_4max_y, color='c')
        plt.xlim(10,40)
        plt.ylim(0,4)
        plt.xlabel("hours post fertilization")
        plt.ylabel("time since last extremum")
        plt.title(f'Mean over {mean_tracks} tracks', position = (0.5, 0.97))
        plt.figtext(
            0.7596, 0.85, 
            f'Minima period: \
                {round(mean_xy_mins["period_deltas"].mean(), 1)} \
                hpf, s.d. {round(mean_xy_mins["period_deltas"].std(), 2)}',
                color='m')
        plt.figtext(
            0.757, 0.83, 
            f'Maxima period: \
                {round(mean_xy_maxs["period_deltas"].mean(), 1)} \
                hpf, s.d. {round(mean_xy_maxs["period_deltas"].std(), 2)}',
                color='c')
        plt.savefig(f"AnalysisResults/{results_path}/\
                    mean_xy_periods{datetimestamp}.png")
        plt.close('4ext')
        
        plt.figure('4_msd', figsize=(18,12))
    
        plot_4_msd_x = dist_mean_stats['common_time_hpf']
        plot_4_msd_y = dist_mean_stats['msd']
    
        plt.plot(plot_4_msd_x, plot_4_msd_y)
        plt.xlim(10, 40)
        plt.ylim(0, 5000)
        plt.xlabel("hours post fertilization")
        plt.ylabel("mean squared displacement ($\mathregular{μm^2}$)")
        plt.title(f'Mean over {mean_tracks} tracks', position = (0.5, 0.97))
        plt.savefig(f"AnalysisResults/{results_path}/\
                    mean_squared_displacement{datetimestamp}.png")
        plt.close('4_msd')
        
        plt.figure(5, figsize=(18,12))
    
        plot_5_x = time_mean_stats['common_time_hpf']
        plot_5_y = time_mean_stats['z_speed_mean']
    
        plt.plot(plot_5_x, plot_5_y)
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['z_speed_min'], color='m')
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['z_speed_max'], color='c')
        plt.xlim(10, 40)
        plt.ylim(-30, 60)
        plt.xlabel("hours post fertilization")
        plt.ylabel("mean z speed in μm/h, rolling 15 min average")
        plt.title(f'Mean over {mean_tracks} tracks', position = (0.5, 0.97))
        plt.figtext(
            0.7596, 0.85, 
            f'Minima period: {round(mean_z_mins["period_deltas"].mean(), 1)} \
                hpf, s.d. {round(mean_z_mins["period_deltas"].std(), 2)}')
        plt.figtext(
            0.757, 0.83, 
            f'Maxima period: {round(mean_z_maxs["period_deltas"].mean(), 1)} \
                hpf, s.d. {round(mean_z_maxs["period_deltas"].std(), 2)}')
        plt.savefig(f"AnalysisResults/{results_path}/\
                    mean_z_speed{datetimestamp}.png")
        plt.close(5)
    
        plt.figure('5ext', figsize=(18,12))
        
        plot_5min_x = mean_z_mins['common_time_hpf']
        plot_5min_y = mean_z_mins['period_deltas']
        plot_5max_x = mean_z_maxs['common_time_hpf']
        plot_5max_y = mean_z_maxs['period_deltas']
        
        plt.plot(plot_5min_x, plot_5min_y, color='m')
        plt.plot(plot_5max_x, plot_5max_y, color='c')
        plt.xlim(10,40)
        plt.ylim(0,4)
        plt.xlabel("hours post fertilization")
        plt.ylabel("time since last extremum")
        plt.title(f'Mean over {mean_tracks} tracks', position = (0.5, 0.97))
        plt.figtext(
            0.7596, 0.85, 
            f'Minima period: {round(mean_z_mins["period_deltas"].mean(), 1)} \
                hpf, s.d. {round(mean_z_mins["period_deltas"].std(), 2)}',
                color='m')
        plt.figtext(
            0.757, 0.83, 
            f'Maxima period: {round(mean_z_maxs["period_deltas"].mean(), 1)} \
                hpf, s.d. {round(mean_z_maxs["period_deltas"].std(), 2)}',
                color='c')
        plt.savefig(f"AnalysisResults/{results_path}/\
                    mean_z_periods{datetimestamp}.png")
        plt.close('5ext')
    
        plt.figure(6, figsize=(18,12))
    
        plot_6_x = time_mean_stats['common_time_hpf']
        plot_6_y = time_mean_stats['persistence_mean']
    
        plt.plot(plot_6_x, plot_6_y)
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['persistence_min'], color='m')
        plt.scatter(time_mean_stats['common_time_hpf'], 
                    time_mean_stats['persistence_max'], color='c')
        plt.xlim(10, 40)
        plt.ylim(0, 1)
        plt.xlabel("hours post fertilization")
        plt.ylabel("mean persistence, rolling 15 min average")
        plt.title(f'Mean over {mean_tracks} tracks', position = (0.5, 0.97))
        plt.figtext(
            0.7596, 0.85, 
            f'Minima period: \
                {round(mean_persistence_mins["period_deltas"].mean(), 1)}'
            f' hpf, s.d. \
                {round(mean_persistence_mins["period_deltas"].std(), 2)}')
        plt.figtext(
            0.757, 0.83, 
            f'Maxima period: \
                {round(mean_persistence_maxs["period_deltas"].mean(), 1)}'
            f' hpf, s.d. \
                {round(mean_persistence_maxs["period_deltas"].std(), 2)}')
        plt.savefig(f"AnalysisResults/{results_path}/\
                    mean_persistence{datetimestamp}.png")
        plt.close(6)
    
        plt.figure('6ext', figsize=(18,12))
        
        plot_6min_x = mean_persistence_mins['common_time_hpf']
        plot_6min_y = mean_persistence_mins['period_deltas']
        plot_6max_x = mean_persistence_maxs['common_time_hpf']
        plot_6max_y = mean_persistence_maxs['period_deltas']
        
        plt.plot(plot_6min_x, plot_6min_y, color='m')
        plt.plot(plot_6max_x, plot_6max_y, color='c')
        plt.xlim(10,40)
        plt.ylim(0,4)
        plt.xlabel("hours post fertilization")
        plt.ylabel("time since last extremum")
        plt.title(f'Mean over {mean_tracks} tracks', position = (0.5, 0.97))
        plt.figtext(
            0.7596, 0.85, 
            f'Minima period: \
                {round(mean_persistence_mins["period_deltas"].mean(), 1)}'
            f' hpf, s.d. \
                {round(mean_persistence_mins["period_deltas"].std(), 2)}',
                color='m')
        plt.figtext(
            0.757, 0.83, 
            f'Maxima period: \
                {round(mean_persistence_maxs["period_deltas"].mean(), 1)}'
            f' hpf, s.d. \
                {round(mean_persistence_maxs["period_deltas"].std(), 2)}',
                color='c')
        plt.savefig(f"AnalysisResults/{results_path}/\
                    mean_persistence_periods{datetimestamp}.png")
        plt.close('6ext')
        
        plt.figure('4_6_overlaid', figsize = (18,12))
        
        plot_common_x = time_mean_stats['common_time_hpf']
        plot_speed_y = time_mean_stats['xy_speed_mean']
        plot_persistence_y = time_mean_stats['persistence_mean'] * 100
        
        plt.plot(plot_common_x, plot_speed_y, color='m')
        plt.plot(plot_common_x, plot_persistence_y, color='c')
        plt.xlim(10,40)
        plt.xlabel("hours post fertilization")
        plt.title(f'Mean over {mean_tracks} tracks', position = (0.5, 0.97))
        plt.figtext(0.7596, 0.85, "speed", color='m')
        plt.figtext(0.757, 0.83, "persistence", color='c')
        plt.savefig(f"AnalysisResults/{results_path}/\
                    speed_persistence{datetimestamp}.png")
        
        plt.close('4_6_overlaid')
    
        print("\n")
        
        # CALCULATE THE PAIRWISE START AND END DISTANCES BETWEEN TRACKS
    
        # The different cases of track times are:
        # 
        # case 1: reference track starts before measured track starts 
        # and reference track ends after measured track ends
        # case 2: ref starts before meas and ref ends before meas
        # case 3: ref starts after meas and ref ends after meas
        # case 4: ref starts after meas and ref ends before meas
        #
        # To match start and end times properly:
        #
        # case 1: ref time should be matched to meas start and ref
        # time should be matched to meas end
        # case 2: ref time should be matched to meas start and meas
        # time should be matched to ref end
        # case 3: meas time should be matched to ref start and ref
        # time should be matched to meas end
        # case 4: meas time should be matched to ref start and meas
        # time should be matched to ref end
    
        plt.figure(7, figsize=(18,12))
    
        fig7_counter = 0
            
        # create an empty list that will be iteratively appended to
        # as the following for loop runs
        list_of_dists = []
    
        list_of_dists_spec = []
    
        # this is used to avoid comparing tracks with themselves 
        # (starting at 1 instead of 0) and with tracks they've already
        # been compared to (incrementing by 1 per 1° loop)
        
        if len(track_anal) > 1:
            counter = 1
        else:
            counter = 0

        # keep count of non-overlapping tracks
        skip_counter = 0
    
        # 1° loop for iterating over reference tracks
        for track in track_anal['track_num']:
            
            # 2° loop for iterating over measured tracks, starting
            # from counter position to the end of the dataframe
            for i in range(counter, len(track_anal)):
                
                # create a 1-column dataframe containing
                # start and end times
                ref_track = track_anal.loc[track_anal["track_num"] == track]
                # convert the 1-column dataframe to a series,
                # as meas_track is
                ref_track = ref_track.iloc[0]
                # full tracking data for the reference frame
                spots_ref_track = spots_sorted.loc[spots_sorted["TRACK_ID"] 
                                                   == track]
                # create a series containing start and end times
                meas_track = track_anal.iloc[i]
                # full tracking data for the measured frame
                spots_meas_track = spots_sorted.loc[spots_sorted["TRACK_ID"] 
                                                    == int(track_anal.iloc[i]
                                                           ["track_num"])]
                
                # setting variables for readability
                ref_x_start = ref_track["x_start"]
                ref_y_start = ref_track["y_start"]
                ref_x_end = ref_track["x_end"]
                ref_y_end = ref_track["y_end"]
                meas_x_start = meas_track["x_start"]
                meas_y_start = meas_track["y_start"]
                meas_x_end = meas_track["x_end"]
                meas_y_end = meas_track["y_end"]
    
        # case 1:
                if float(
                        ref_track["t_start"]) <= float(
                            meas_track["t_start"]) and float(
                                ref_track["t_end"]) >= float(
                                    meas_track["t_end"]):
                           
                    ref_index_pos_at_meas_start_time_c1 = (
                        spots_ref_track.index[spots_ref_track["POSITION_T"] 
                                              == meas_track["t_start"]])
                    ref_index_pos_at_meas_end_time_c1 = (
                        spots_ref_track.index[spots_ref_track["POSITION_T"] 
                                              == meas_track["t_end"]])
                    
                    ref_x_start = (spots_ref_track.
                                   loc[ref_index_pos_at_meas_start_time_c1][
                                       "POSITION_X"])
                    ref_y_start = (spots_ref_track.
                                   loc[ref_index_pos_at_meas_start_time_c1][
                                       "POSITION_Y"])
                    ref_x_end = (spots_ref_track.
                                 loc[ref_index_pos_at_meas_end_time_c1][
                                     "POSITION_X"])
                    ref_y_end = (spots_ref_track.
                                 loc[ref_index_pos_at_meas_end_time_c1][
                                     "POSITION_Y"])
                    
                    start_time = meas_track["t_start"]
                    end_time = meas_track["t_end"]
                     
                    if detailed_log == 'y':
                        log = open(f"AnalysisResults/{results_path}/\
                                   distance_log{datetimestamp}.txt", "a+")
                        log.write(f'Track {track} and track \
                                  {int(meas_track["track_num"])} \
                                      are in case 1')
                        log.write("\n")
                        log.close()
                    
    
        # case 2:
                elif float(
                        ref_track["t_start"]) <= float(
                            meas_track["t_start"]) and float(
                                ref_track["t_end"]) <= float(
                                    meas_track["t_end"]) and float(
                                        ref_track["t_end"]) >= float(
                                            meas_track["t_start"]):
                          
                    ref_index_pos_at_meas_start_time_c2 = (
                        spots_ref_track.index[spots_ref_track["POSITION_T"] 
                                              == meas_track["t_start"]])      
                    meas_index_pos_at_ref_end_time_c2 = (
                        spots_meas_track.index[spots_meas_track["POSITION_T"] 
                                               == ref_track["t_end"]])
                    
                    ref_x_start = (spots_ref_track.
                                   loc[ref_index_pos_at_meas_start_time_c2][
                                       "POSITION_X"])
                    ref_y_start = (spots_ref_track.
                                   loc[ref_index_pos_at_meas_start_time_c2][
                                       "POSITION_Y"])
                    meas_x_end = (spots_meas_track.
                                  loc[meas_index_pos_at_ref_end_time_c2][
                                      "POSITION_X"])
                    meas_y_end = (spots_meas_track.
                                  loc[meas_index_pos_at_ref_end_time_c2][
                                      "POSITION_Y"])
    
                    start_time = meas_track["t_start"]
                    end_time = ref_track["t_end"]
                    
                    if detailed_log == 'y':
                        log = open(f"AnalysisResults/{results_path}/\
                                   distance_log.txt{datetimestamp}", "a+")
                        log.write(f'Track {track} and track \
                                  {int(meas_track["track_num"])} \
                                      are in case 2')
                        log.write("\n")
                        log.close()
                    
        # case 3:
                elif float(
                        ref_track["t_start"]) >= float(
                            meas_track["t_start"]) and float(
                                ref_track["t_end"]) >= float(
                                    meas_track["t_end"]) and float(
                                        ref_track["t_start"]) <= float(
                                            meas_track["t_end"]):
                  
                    meas_index_pos_at_ref_start_time_c3 = (
                        spots_meas_track.index[spots_meas_track["POSITION_T"] 
                                               == ref_track["t_start"]])
                    ref_index_pos_at_meas_end_time_c3 = (
                        spots_ref_track.index[spots_ref_track ["POSITION_T"] 
                                              == meas_track["t_end"]])
                    
                    ref_x_end = (spots_ref_track.
                                 loc[ref_index_pos_at_meas_end_time_c3][
                                     "POSITION_X"])
                    ref_y_end = (spots_ref_track.
                                 loc[ref_index_pos_at_meas_end_time_c3][
                                     "POSITION_Y"])
                    meas_x_start = (spots_meas_track.
                                    loc[meas_index_pos_at_ref_start_time_c3][
                                        "POSITION_X"])
                    meas_y_start = (spots_meas_track.
                                    loc[meas_index_pos_at_ref_start_time_c3][
                                        "POSITION_Y"])
               
                    start_time = ref_track["t_start"]
                    end_time = meas_track["t_end"]
                   
                    if detailed_log == 'y':
                        log = open(f"AnalysisResults/{results_path}/\
                                   distance_log{datetimestamp}.txt", "a+")
                        log.write(f'Track {track} and track \
                                  {int(meas_track["track_num"])} \
                                      are in case 3')
                        log.write("\n")
                        log.close()
    
        # case 4:
                elif float(
                        ref_track["t_start"]) >= float(
                            meas_track["t_start"]) and float(
                                ref_track["t_end"]) <= float(
                                    meas_track["t_end"]):
                              
                    meas_index_pos_at_ref_start_time_c4 = (
                        spots_meas_track.index[spots_meas_track["POSITION_T"] 
                                               == ref_track["t_start"]])
                    meas_index_pos_at_ref_end_time_c4 = (
                        spots_meas_track.index[spots_meas_track["POSITION_T"] 
                                               == ref_track["t_end"]])
                    
                    meas_x_start = (spots_meas_track.
                                    loc[meas_index_pos_at_ref_start_time_c4][
                                        "POSITION_X"])
                    meas_y_start = (spots_meas_track.
                                    loc[meas_index_pos_at_ref_start_time_c4][
                                        "POSITION_Y"])
                    meas_x_end = (spots_meas_track.
                                  loc[meas_index_pos_at_ref_end_time_c4][
                                      "POSITION_X"])
                    meas_y_end = (spots_meas_track.
                                  loc[meas_index_pos_at_ref_end_time_c4][
                                      "POSITION_Y"])
                  
                    start_time = ref_track["t_start"]
                    end_time = ref_track["t_end"]
                    
                    if detailed_log == 'y':
                        log = open(f"AnalysisResults/{results_path}/\
                                   distance_log{datetimestamp}.txt", "a+")
                        log.write(f'Track {track} and track \
                                  {int(meas_track["track_num"])} \
                                      are in case 4')
                        log.write("\n")
                        log.close()
    
        # when there is no overlap, skip the pair:            
                else:
                    skip_counter += 1
                    if detailed_log == 'y':
                        log = open(f"AnalysisResults/{results_path}/\
                                   distance_log{datetimestamp}.txt", "a+")
                        log.write(f'Track {track} and track \
                                  {int(meas_track["track_num"])} \
                                      do not overlap')
                        log.write("\n")
                        log.close()
                    # skip outputting anything 
                    # when there is no overlap
                    continue
          
                start_dist = dist2d(ref_x_start, ref_y_start, 
                                    meas_x_start, meas_y_start)
                end_dist = dist2d(ref_x_end, ref_y_end, 
                                  meas_x_end, meas_y_end)
                
                ref_slope = slope(ref_x_start, ref_y_start, 
                                  ref_x_end, ref_y_end)
                meas_slope = slope(meas_x_start, meas_y_start, 
                                   meas_x_end, meas_y_end)
    
                tracks_ang = angle(ref_slope, meas_slope)     
                ang_pair_correl = math.cos(math.radians(tracks_ang))
                
                # create a dictionary
                new_dist = {
                    'reference_track':track,
                    'measured_track':int(meas_track["track_num"]),
                    'dev_start_time(hpf)':round(((start_time / 3600) 
                                                 + dev_time), 2),
                    'dev_end_time(hpf)':round(((end_time / 3600) 
                                               + dev_time), 2),
                    'time_delta(hpf)':round((end_time / 3600) 
                                            - (start_time / 3600), 2),
                    'start_distance(μm)':round(start_dist, 2),
                    'end_distance(μm)':round(end_dist, 2),
                    'dist_delta(μm)':round(end_dist - start_dist, 2),
                    'angle(deg)':round(tracks_ang, 2),
                    'angle_pair_correl':round(ang_pair_correl, 5)
                    }
                
                fig7_counter += 1
                
                if specific_pairwise == 'y':
                
                    new_dist_spec = {
                        'reference_track':track,
                        'measured_track':int(meas_track["track_num"]),
                        'dev_start_time(hpf)':round(((start_time / 3600) 
                                                     + dev_time)),
                        'dev_end_time(hpf)':round(((end_time / 3600) 
                                                   + dev_time)),
                        'time_delta(hpf)':round((end_time / 3600) 
                                                - (start_time / 3600)),
                        'start_distance(μm)':round(start_dist, 2),
                        'end_distance(μm)':round(end_dist, 2),
                        'dist_delta(μm)':round(end_dist - start_dist, 2),
                        'angle(deg)':round(tracks_ang, 2),
                        'angle_pair_correl':round(ang_pair_correl, 5)
                        } # creates a dictionary
                    
                    list_of_dists_spec.append(new_dist_spec)
                  
                # limited to 4 as too many tracks will clutter a graph
                # into illegibility       
                if fig7_counter <= 20 and new_dist["time_delta(hpf)"] >= 4:
                        
                    plot_7_x = [new_dist['dev_start_time(hpf)'], new_dist[
                        'dev_end_time(hpf)']]
                    plot_7_y = [new_dist['start_distance(μm)'], new_dist[
                        'end_distance(μm)']]
                    
                    plt.plot(plot_7_x, plot_7_y, label = f'{track} - \
                             {int(meas_track["track_num"])}')
                    plt.legend(ncol = 2)
                    
                # make list_of_dists a list of dictionaries
                list_of_dists.append(new_dist)
    
            # increment counter to skip pairwise comparisons
            # that have already been made
            counter += 1
            
            # print a countdown counter to the shell
            if len(track_anal) - (counter - 2) > 1:
                print(f'{len(track_anal) - (counter - 2)} tracks remaining')
            else:
                print("1 track remaining")
                
        print("\n")       
        
        # create a dataframe from list_of_dicts
        track_dists = pd.DataFrame(list_of_dists)
    
        plt.xlabel("hours post fertilization")
        plt.ylabel("distance in μm")
        plt.xlim(10, 40)
        plt.ylim(0, 150)
        plt.savefig(f"AnalysisResults/{results_path}/\
                    distances{datetimestamp}.png")
        plt.close(7)
        
        linear_regressor = LinearRegression()
        linear_regressor.fit(track_dists['start_distance(μm)'].
                             values.reshape(-1, 1), track_dists[
                                 'angle_pair_correl'].values.reshape(-1, 1))
        dist_angle_pred = linear_regressor.predict(
            track_dists['start_distance(μm)'].values.reshape(-1, 1))
        dist_angle_r = round(((r2_score(track_dists['angle_pair_correl'].
                                        values.reshape(-1, 1), 
                                        dist_angle_pred)) ** 0.5), 2)
        
        
        # commenting out polynomial regression as it's not needed
        # now, but leaving it in in case it's needed
        
        # poly_2_regressor = PolynomialFeatures(degree = 2)
        # poly_3_regressor = PolynomialFeatures(degree = 3)
        
        # start_dist_poly_2 = poly_2_regressor.fit_transform(
        #     track_dists['start_distance(μm)'].values.reshape(-1, 1))
        # start_dist_poly_3 = poly_3_regressor.fit_transform(
        #     track_dists['start_distance(μm)'].values.reshape(-1, 1))
        
        # poly_2_regressor = LinearRegression()
        # poly_2_regressor.fit(start_dist_poly_2, track_dists[
        #     'angle_pair_correl'].values.reshape(-1, 1))
        # dist_angle_poly_2_pred = poly_2_regressor.predict(start_dist_poly_2)
        
        # poly_3_regressor = LinearRegression()
        # poly_3_regressor.fit(start_dist_poly_3, track_dists[
        #     'angle_pair_correl'].values.reshape(-1, 1))
        # dist_angle_poly_3_pred = poly_3_regressor.predict(start_dist_poly_3)

        
        plt.figure('7_ang', figsize=(18,12))
        
        plt.scatter(track_dists['start_distance(μm)'], 
                    track_dists['angle_pair_correl'], color='k')
        plt.plot(track_dists['start_distance(μm)'].
                 values.reshape(-1, 1), dist_angle_pred, color='r')
        # commenting out polynomial regression graphs
        # plt.scatter(track_dists['start_distance(μm)'].values.reshape(-1, 1), 
        #             dist_angle_poly_2_pred, color='darkorange')
        # plt.scatter(track_dists['start_distance(μm)'].values.reshape(-1, 1), 
        #             dist_angle_poly_3_pred, color='red')
        
        plt.xlabel("Starting distance between cells (μm)")
        plt.ylabel("cosine of trajectory angle")
                
        plt.title(dist_angle_r, position = (0.5, 0.97))
        
        plt.savefig(f"AnalysisResults/{results_path}/\
                    dist_angle_scatter{datetimestamp}.png")
        
        plt.close('7_ang')
            
        if automatic_specific_tracker > 1:
            
            plt.figure(8, figsize=(18,12))
    
            fig8_counter = 0
            
            # match start and end times before comparisons
            list_of_spec_dists = []
    
            # avoids comparing tracks with themselves (starting at 1 
            # instead of 0) and with tracks they've already been 
            # compared to (incrementing by 1 per 1° loop)
            counter_spec = 1
    
            # 1° loop for iterating over reference tracks
            for track in track_anal_spec['track_spec_num']:
                
                # 2° loop for iterating over measured tracks, starting
                # from counter position to the end of the dataframe
                for i in range(counter_spec, len(track_anal_spec)):
                    
                    # create a 1-column dataframe containing
                    # start and end times
                    ref_spec_track = track_anal_spec.loc[track_anal_spec[
                        "track_spec_num"] == track] 
                    # convert the 1-column dataframe to a series,
                    # as meas_track is
                    ref_spec_track = ref_spec_track.iloc[0]
                    # create a series containing start and end times
                    meas_spec_track = track_anal_spec.iloc[i]
    
                    # setting variables for readability
                    ref_spec_x_start = ref_spec_track["x_spec_start"]
                    ref_spec_y_start = ref_spec_track["y_spec_start"]
                    ref_spec_x_end = ref_spec_track["x_spec_end"]
                    ref_spec_y_end = ref_spec_track["y_spec_end"]
                    meas_spec_x_start = meas_spec_track["x_spec_start"]
                    meas_spec_y_start = meas_spec_track["y_spec_start"]
                    meas_spec_x_end = meas_spec_track["x_spec_end"]
                    meas_spec_y_end = meas_spec_track["y_spec_end"]
    
    
                    start_spec_time = ref_spec_track["t_spec_start"]
                    end_spec_time = ref_spec_track["t_spec_end"]
    
                    start_spec_dist = dist2d(
                        ref_spec_x_start, ref_spec_y_start, 
                        meas_spec_x_start, meas_spec_y_start)
                    end_spec_dist = dist2d(
                        ref_spec_x_end, ref_spec_y_end, 
                        meas_spec_x_end, meas_spec_y_end)
                    
                    ref_spec_slope = slope(
                        ref_spec_x_start, ref_spec_y_start, 
                        ref_spec_x_end, ref_spec_y_end)
                    meas_spec_slope = slope(
                        meas_spec_x_start, meas_spec_y_start, 
                        meas_spec_x_end, meas_spec_y_end)
    
                    tracks_spec_ang = angle(ref_spec_slope, meas_spec_slope)     
                    ang_spec_pair_correl = math.cos(math.radians(
                        tracks_spec_ang))
    
                    new_spec_dist = {
                        'reference_spec_track':track,
                        'measured_spec_track':int(meas_spec_track[
                            "track_spec_num"]),
                        'dev_spec_start_time(hpf)':round(((start_spec_time 
                                                           / 3600) 
                                                          + dev_time), 2),
                        'rounded_start(hpf)':round(((start_spec_time 
                                                     / 3600) + dev_time)),
                        'dev_spec_end_time(hpf)':round(((end_spec_time 
                                                         / 3600) 
                                                        + dev_time), 2),
                        'rounded_end(hpf)':round(((end_spec_time 
                                                   / 3600) + dev_time)),
                        'time_spec_delta(hpf)':round((end_spec_time 
                                                      / 3600) 
                                                     - (start_spec_time 
                                                        / 3600), 2),
                        'start_spec_distance(μm)':round(start_spec_dist, 2),
                        'end_spec_distance(μm)':round(end_spec_dist, 2),
                        'dist_spec_delta(μm)':round(end_spec_dist 
                                                    - start_spec_dist, 2),
                        'angle_spec(deg)':round(tracks_spec_ang, 2),
                        'angle_spec_pair_correl':round(ang_spec_pair_correl, 
                                                       5)
                        } # creates a dictionary
                    
                    
                    list_of_spec_dists.append(new_spec_dist)
                    
                    fig8_counter += 1
                    
                    # too many tracks will clutter a graph into
                    # illegibility
                    if fig8_counter <= 20:

                        plot_8_x = [new_spec_dist['dev_spec_start_time(hpf)'], 
                                    new_spec_dist['dev_spec_end_time(hpf)']]
                        plot_8_y = [new_spec_dist['start_spec_distance(μm)'], 
                                    new_spec_dist['end_spec_distance(μm)']]
                     
                        plt.plot(plot_8_x, plot_8_y, label = f'{track} - \
                                 {int(meas_spec_track["track_spec_num"])}')
                        plt.legend(ncol = 2)
                        
                # increment counter to skip pairwise comparisons
                # that have already been made
                counter_spec += 1
                    
                # print a countdown counter to the shell
                if len(track_anal_spec) - (counter_spec - 2) > 1:
                    print(f'{len(track_anal_spec) - (counter_spec - 2)} \
                          tracks with specified times remaining')
                else:
                    print("1 track with specified times remaining\n")
    
    
            track_spec_dists = pd.DataFrame(list_of_spec_dists)
            
            plt.xlabel("hours post fertilization")
            plt.ylabel("distance in μm")
            plt.savefig(f"AnalysisResults/{results_path}/\
                        specified_distances{datetimestamp}.png")
            plt.close(8)
    
    
        # output csv files for use in Excel, skipping index column
        
        # utf-8-sig encoding (instead of the utf-8 default) 
        # lets Excel read Greek characters
    
        # note that for making selections inside dataframes, '&' is
        # used not as a bitwise but as a boolean operator, while 
        # 'and' cannot be used for this purpose ('&' can be 
        # overloaded but 'and' is hardcoded and series/dataframes 
        # can't be booleans as a whole)
    
        time_counts.to_csv(
            f'AnalysisResults/{results_path}/\
                time_counts{datetimestamp}.csv', 
                encoding = 'utf-8-sig')    
        track_anal.to_csv(
            f'AnalysisResults/{results_path}/\
                tracks_analyzed{datetimestamp}.csv', 
                index = False, encoding = 'utf-8-sig')
        track_anal_spec.to_csv(
            f'AnalysisResults/{results_path}/\
                tracks_analyzed_specified{datetimestamp}.csv', 
                index = False, encoding = 'utf-8-sig')
        track_dists.to_csv(
            f'AnalysisResults/{results_path}/\
                track_distances{datetimestamp}.csv', 
                index = False, encoding = 'utf-8-sig')
        spots_rolling.to_csv(
            f'AnalysisResults/{results_path}/\
                rolling_statistics{datetimestamp}.csv', 
                index = False, encoding = 'utf-8-sig')
        spots_sorted_good.to_csv(
            f'AnalysisResults/{results_path}/\
                spots_sorted_filtered{datetimestamp}.csv', 
                index = False, encoding = 'utf-8-sig')
        time_mean_stats.to_csv(
            f'AnalysisResults/{results_path}/\
                mean_stats_over_time{datetimestamp}.csv', 
                index = False, encoding = 'utf-8-sig')  
        distance_stats.to_csv(
            f'AnalysisResults/{results_path}/\
                distance_stats{datetimestamp}.csv', 
                index = False, encoding = 'utf-8-sig')
        dist_mean_stats.to_csv(
            f'AnalysisResults/{results_path}/\
                distance_mean_stats{datetimestamp}.csv', 
                index = False, encoding = 'utf-8-sig')
        # only keep tracks with more than 4 hours common time
        track_dists[track_dists["time_delta(hpf)"] >= 4].to_csv(
            f'AnalysisResults/{results_path}/\
                track_distance_filtered_4h{datetimestamp}.csv', 
                index = False, encoding = 'utf-8-sig')
        track_dists[(track_dists["start_distance(μm)"] <= 20) & (
            track_dists["time_delta(hpf)"] >= 4)].to_csv(
                f'AnalysisResults/{results_path}/\
                    track_distance_filtered_20um{datetimestamp}.csv', 
                    index = False, encoding = 'utf-8-sig')
        track_dists[(track_dists["start_distance(μm)"] > 20) & (
            track_dists["start_distance(μm)"] <= 50) & (
                track_dists["time_delta(hpf)"] >= 4)].to_csv(
                    f'AnalysisResults/{results_path}/\
                        track_distance_filtered_20_50um{datetimestamp}.csv', 
                        index = False, encoding = 'utf-8-sig')
        track_dists[(track_dists["start_distance(μm)"] > 50) & (
            track_dists["time_delta(hpf)"] >= 4)].to_csv(
                f'AnalysisResults/{results_path}/\
                    track_distance_filtered_over50um{datetimestamp}.csv', 
                    index = False, encoding = 'utf-8-sig')
    
        if automatic_specific_tracker > 1:
            track_spec_dists.to_csv(
                f'AnalysisResults/{results_path}/specified_pairwise_\
                    {specific_pairwise_start}-{specific_pairwise_end}\
                        {datetimestamp}hpf.csv', 
                        index = False, encoding = 'utf-8-sig')
    
        filtered_correl_check = False
        start_dist_20_correl_check = False
        start_dist_20_50_correl_check = False
        start_dist_over_50_correl_check = False
    
    
        if len(track_dists[track_dists[
                "time_delta(hpf)"] >= 4]['start_distance(μm)']) > 2:
            filtered_correl = track_dists[track_dists[
                "time_delta(hpf)"] >= 4]['start_distance(μm)'].corr(
                    track_dists[track_dists["time_delta(hpf)"] >= 4][
                        'end_distance(μm)'])
            filtered_correl_check = True
    
        if len(track_dists[(track_dists[
                "start_distance(μm)"] <= 20) & (track_dists[
                    "time_delta(hpf)"] >= 4)]) > 2:
            start_dist_20_correl = track_dists[(track_dists[
                "start_distance(μm)"] <= 20) & (track_dists[
                    "time_delta(hpf)"] >= 4)]['start_distance(μm)'].corr(
                        track_dists[(track_dists["start_distance(μm)"] <= 20) 
                                    & (track_dists["time_delta(hpf)"] >= 4)]
                        ['end_distance(μm)'])
            start_dist_20_correl_check = True
    
        if len(track_dists[(track_dists[
                "start_distance(μm)"] > 20) & (track_dists[
                    "start_distance(μm)"] <= 50) & (
                        track_dists["time_delta(hpf)"] >= 4)]) > 2:
            start_dist_20_50_correl = track_dists[
                (track_dists["start_distance(μm)"] > 20) & (
                    track_dists["start_distance(μm)"] <= 50) & (
                        track_dists["time_delta(hpf)"] >= 4)][
                            'start_distance(μm)'].corr(track_dists[(
                                track_dists["start_distance(μm)"] > 20) & (
                                    track_dists[
                                        "start_distance(μm)"] <= 50) & (
                                            track_dists[
                                                "time_delta(hpf)"] >= 4)][
                                                    'end_distance(μm)'])
            start_dist_20_50_correl_check = True
    
        if len(track_dists[(track_dists[
                "start_distance(μm)"] > 50) & (
                    track_dists["time_delta(hpf)"] >= 4)]) > 2:
            start_dist_over_50_correl = track_dists[
                (track_dists["start_distance(μm)"] > 50) & (
                    track_dists["time_delta(hpf)"] >= 4)][
                        'start_distance(μm)'].corr(track_dists[(
                            track_dists["start_distance(μm)"] > 50) & (
                                track_dists["time_delta(hpf)"] >= 4)][
                                    'end_distance(μm)'])
            start_dist_over_50_correl_check = True
    
   
        if automatic_specific_tracker > 2:
            pairwise_specific_correl = track_spec_dists[
                'start_spec_distance(μm)'].corr(track_spec_dists[
                    'end_spec_distance(μm)'])
            
    
        log = open(f'AnalysisResults/{results_path}/\
                   distance_log{datetimestamp}.txt', "a+", 
                   encoding = 'utf-8-sig')
        if detailed_log == 'y':
            log.write("\n")
        log.write(f'Working directory: {work_dir}.')
        log.write("\n\n")
        log.write(f'Region analyzed: {outline_region}.')
        log.write("\n\n")
        log.write(f'Starting developmental time: {dev_time} hpf.')
        log.write("\n\n")
        log.write(f'Time step is {time_step} seconds and there are \
                  {steps_15_min} steps for every 15 minutes.')
        log.write("\n\n")
        if len(spots_rolling["track"].unique()) > 1:
            log.write(f'{len(spots_rolling["track"].unique())} \
                      tracks analyzed.')
        else:
            log.write(f'{len(spots_rolling["track"].unique())} \
                      track analyzed.')
        log.write("\n\n")
        log.write(f'{len(track_dists)} pairwise comparisons made between \
                  {len(track_anal)} tracks, with \
                      {skip_counter} tracks skipped.')
        log.write("\n\n")
        log.write(f'{len(track_dists[track_dists["time_delta(hpf)"] >= 4])} \
                  pairwise comparisons retained for the \
                      filtered distance list.')
        log.write("\n\n")
        log.write(f'The mean minima period for xy speed is \
                  {round(mean_xy_mins["period_deltas"].mean(), 1)} \
                      with a standard deviation of \
                          {round(mean_xy_mins["period_deltas"].std(), 2)}.')
        log.write("\n\n")
        log.write(f'The mean maxima period for xy speed is \
                  {round(mean_xy_maxs["period_deltas"].mean(), 1)} \
                      with a standard deviation of \
                          {round(mean_xy_maxs["period_deltas"].std(), 2)}.')
        log.write("\n\n")
        log.write(f'The mean minima period for z speed is \
                  {round(mean_z_mins["period_deltas"].mean(), 1)} \
                      with a standard deviation of \
                          {round(mean_z_mins["period_deltas"].std(), 2)}.')
        log.write("\n\n")
        log.write(f'The mean maxima period for z speed is \
                  {round(mean_z_maxs["period_deltas"].mean(), 1)} \
                      with a standard deviation of \
                          {round(mean_z_maxs["period_deltas"].std(), 2)}.')
        log.write("\n\n")
        log.write(f'The mean minima period for persistence is \
                  {round(mean_persistence_mins["period_deltas"].mean(), 1)}'
                  f' with a standard deviation of '
                  f'{round(mean_persistence_mins["period_deltas"].std(), 2)}.'
                  )
        log.write("\n\n")
        log.write(f'The mean maxima period for persistence is \
                  {round(mean_persistence_maxs["period_deltas"].mean(), 1)}'
                  f' with a standard deviation of '
                  f'{round(mean_persistence_maxs["period_deltas"].std(), 2)}.'
                  )
        log.write("\n\n")
        if filtered_correl_check == True:
            log.write(f'The Pearson correlation between start and end \
                      distances for tracks that have at least 4 hours in \
                          common is {round(filtered_correl, 3)}.')
            log.write("\n\n")
        else:
            log.write("Not enough tracks to calculate Pearson correlation \
                      between start and end distances for tracks that have \
                          at least 4 hours in common.")
            log.write("\n\n")
        if automatic_specific_tracker > 2:
            log.write(f'The Pearson correlation between start and end \
                      distances for the {automatic_specific_tracker} tracks \
                          with times that match {specific_pairwise_start} \
                              and {specific_pairwise_end} hpf is \
                                  {round(pairwise_specific_correl, 3)}.')
            log.write("\n\n")
        else:
            log.write(f'Not enough tracks to calculate Pearson correlation \
                      between start and end distances for tracks with times \
                          that match {specific_pairwise_start} and \
                              {specific_pairwise_end} hpf.')
            log.write("\n\n")    
        if start_dist_20_correl_check == True:
            log.write(f'The Pearson correlation between start and end \
                      distances for filtered tracks that have at least 4 \
                          hours in common and start within 20 μm is \
                              {round(start_dist_20_correl, 3)}.')
            log.write("\n\n")
        else:
            log.write("Not enough tracks to calculate Pearson correlation \
                      between start and end distances for filtered tracks \
                          that have at least 4 hours in common and start \
                              within 20 μm.")
            log.write("\n\n")
        if start_dist_20_50_correl_check == True:
            log.write(f'The Pearson correlation between start and end \
                      distances for filtered tracks that have at least 4 \
                          hours in common and start within 20 - 50 μm is \
                              {round(start_dist_20_50_correl, 3)}.')
            log.write("\n\n")
        else:
            log.write("Not enough tracks to calculate Pearson correlation \
                      between start and end distances for filtered tracks \
                          that have at least 4 hours in common and start \
                              within 20 - 50 μm.")
            log.write("\n\n")
        if start_dist_over_50_correl_check == True:
            log.write(f'The Pearson correlation between start and end \
                      distances for filtered tracks that have at least 4 \
                          hours in common and start farther than 50 μm is \
                              {round(start_dist_over_50_correl, 3)}.')
            log.write("\n\n")
        else:
            log.write("Not enough tracks to calculate Pearson correlation \
                      between start and end distances for filtered tracks \
                          that have at least 4 hours in common and start \
                              farther than 50 μm.")
            log.write("\n\n")
        log.write(f'The script took {datetime.now() - script_start_time} \
                  to run.')
        log.close()

    else:
        log = open(f'AnalysisResults/{results_path}/\
                   distance_log{datetimestamp}.txt', "a+", 
                   encoding = 'utf-8-sig')
        log.write(f'Working directory: {work_dir}.')
        log.write("\n\n")
        log.write(f'Region analyzed: {outline_region}.')
        log.write("\n\n")
        log.write("No tracks to analyze in this region.")
        log.write("\n\n")
        log.write(f'The script took {datetime.now() - script_start_time} \
                  to run.')
        log.close()
        

    print("\n")    
    print(f'Run time: {datetime.now() - script_start_time}')
    print("\n")


##############################################################################

if __name__ == '__main__':

    # to analyze TrackMate/Assisted Nuclei 3D tracking spreadsheets
    # as dataframes
    import pandas as pd

    # for file handling
    import os

    # to handle exits
    import sys
    
    folder_with_tracks = os.getcwd()
    
    anim_check = input(
        "Create animations? This adds a significant amount of time. ")
    
    arch_results = 'WholeArch'
    dorsal_results = 'DorsalArch'
    intermediate_results = 'IntermediateArch'
    ventral_results = 'VentralArch'
    anterior_results = 'AnteriorArch'
    posterior_results = 'PosteriorArch'
    image_results = 'WholeImage'
            
    try:
        # the tracking plugin generates a tab-delimited
        # text file with a .xls extension
        spots_file = pd.read_csv(
            'Spots in tracks statistics.xls', sep = '\t', engine='python')
        goodtracks_file = pd.read_csv(
            'Tracks Statistics.xls', sep = '\t', engine='python')
        outline_file = pd.read_csv(
            'FirstOutline.xls', sep = '\t', engine='python')
        dorsal_outline_file = pd.read_csv(
            'FirstDorsalOutline.xls', sep = '\t', engine='python')
        intermediate_outline_file = pd.read_csv(
            'FirstIntermediateOutline.xls', sep = '\t', engine='python')
        ventral_outline_file = pd.read_csv(
            'FirstVentralOutline.xls', sep = '\t', engine='python')
        anterior_outline_file = pd.read_csv(
            'FirstAnteriorOutline.xls', sep = '\t', engine='python')
        posterior_outline_file = pd.read_csv(
            'FirstPosteriorOutline.xls', sep = '\t', engine='python')
        image_outline_file = pd.read_csv(
            'ImageOutline.xls', sep = '\t', engine='python')
    except:
        print("Make sure Spots in tracks statistics.xls, \
              Tracks Statistics.xls, and all six outline files are present \
                  in this folder and run the script again.")
        sys.exit("File not found")
        
    # get the developmental time at imaging start as an input
    # and check to see if it's valid    
    check_dev_time = False

    while not check_dev_time:
        
        try:
            print("\n")
            developmental_time = float(input(
                "What is the developmental stage (in hpf) of this embryo? "))
            if developmental_time >=0:
                check_dev_time = True
                print("\n")
            else:
                print("\nThat's not a valid developmental stage.\n")
        except:
            print("\nThat's not a valid developmental stage.\n")


    tracks_analysis(folder_with_tracks, spots_file, goodtracks_file, 
                    outline_file, image_outline_file, 
                    developmental_time, arch_results, 
                    'first arch', anim_check)
    # tracks_analysis(folder_with_tracks, spots_file, goodtracks_file, 
    #                 dorsal_outline_file, image_outline_file, 
    #                 developmental_time, dorsal_results, 
    #                 'dorsal first arch', anim_check)
    # tracks_analysis(folder_with_tracks, spots_file, goodtracks_file, 
    #                 intermediate_outline_file, image_outline_file, 
    #                 developmental_time, intermediate_results, 
    #                 'intermediate first arch', anim_check)
    # tracks_analysis(folder_with_tracks, spots_file, goodtracks_file, 
    #                 ventral_outline_file, image_outline_file, 
    #                 developmental_time, ventral_results, 
    #                 'ventral first arch', anim_check)
    # tracks_analysis(folder_with_tracks, spots_file, goodtracks_file, 
    #                 anterior_outline_file, image_outline_file, 
    #                 developmental_time, anterior_results, 
    #                 'anterior first arch', anim_check)
    # tracks_analysis(folder_with_tracks, spots_file, goodtracks_file, 
    #                 posterior_outline_file, image_outline_file, 
    #                 developmental_time, posterior_results, 
    #                 'posterior first arch', anim_check)
    # tracks_analysis(folder_with_tracks, spots_file, goodtracks_file, 
    #                 image_outline_file, image_outline_file, 
    #                 developmental_time, image_results, 
    #                 'whole image', anim_check)

## END