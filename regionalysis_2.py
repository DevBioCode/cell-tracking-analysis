#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:16:03 2020

@author: praveersharma
"""

def regional_start_positions(tracks_folder, image_outline):
    
    from datetime import datetime
    datetimestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    import pandas as pd
    import os
    # import shutil
    import matplotlib.pyplot as plt
    import glob
    
    
    os.chdir(tracks_folder)
    os.chdir('AnalysisResults')
    
    try:
        dorsal_tracks = pd.read_csv(
            glob.glob('DorsalArch/tracks_analyzed*.csv')[0])
    except:
        dorsal_tracks = []
    try:
        intermediate_tracks = pd.read_csv(
            glob.glob('IntermediateArch/tracks_analyzed*.csv')[0])
    except:
        intermediate_tracks = []
    try:
        ventral_tracks = pd.read_csv(
            glob.glob('VentralArch/tracks_analyzed*.csv')[0])
    except:
        ventral_tracks = []
    
    try:
        anterior_tracks = pd.read_csv(
            glob.glob('AnteriorArch/tracks_analyzed*.csv')[0])
    except:
        anterior_tracks = []
    try:
        posterior_tracks = pd.read_csv(
            glob.glob('PosteriorArch/tracks_analyzed*.csv')[0])
    except:
        posterior_tracks = []
        
    plt.figure('DIV', figsize=(18,12))
    
    for i in range(len(dorsal_tracks)):
        plt.scatter(
            dorsal_tracks["x_start"][i], 
            - (dorsal_tracks["y_start"][i]), s = 100, c = 'gold')
    
    for i in range(len(intermediate_tracks)):
        plt.scatter(
            intermediate_tracks["x_start"][i], 
            - (intermediate_tracks["y_start"][i]), s = 100, c 
            = 'cornflowerblue')
        
    for i in range(len(ventral_tracks)):
        plt.scatter(
            ventral_tracks["x_start"][i], 
            - (ventral_tracks["y_start"][i]), s = 100, c = 'deeppink')
        
    plt.scatter(-100, 100, s = 100, c = 'gold', label = 'Dorsal')
    plt.scatter(-100, 100, s = 100, c = 'cornflowerblue', label 
                = 'Intermediate')
    plt.scatter(-100, 100, s = 100, c = 'deeppink', label = 'Ventral')
    
    plt.xlim(0, image_outline["X"][2])
    plt.ylim(-image_outline["Y"][2],0)
    plt.xlabel("distance in μm")
    plt.ylabel("distance in μm")
    plt.legend()

    plt.savefig(f'DIV{datetimestamp}.png')
    plt.close('DIV')
    
    plt.figure('AP', figsize=(18,12))
    
    for i in range(len(anterior_tracks)):
        plt.scatter(anterior_tracks["x_start"][i], 
                    - (anterior_tracks["y_start"][i]), s = 100, c 
                    = 'steelblue')
    
    for i in range(len(posterior_tracks)):
        plt.scatter(posterior_tracks["x_start"][i], 
                    - (posterior_tracks["y_start"][i]), s = 100, c 
                    = 'forestgreen')
       
    plt.scatter(-100, 100, s = 100, c = 'steelblue', label = 'Anterior')
    plt.scatter(-100, 100, s = 100, c = 'forestgreen', label = 'Posterior')
    
    plt.xlim(0, image_outline["X"][2])
    plt.ylim(-image_outline["Y"][2],0)
    plt.xlabel("distance in μm")
    plt.ylabel("distance in μm")
    plt.legend()
    # AP_leg = AP.get_legend()
    # AP_leg.legendHandles[0].set_color('steelblue')
    # AP_leg.legendHandles[1].set_color('forestgreen')
    plt.savefig(f'AP{datetimestamp}.png')
    plt.close('AP')
    

##############################################################################

if __name__ == '__main__':
    
    import pandas as pd
    import os
    
    folder_with_tracks = os.getcwd()
    image_outline_file = pd.read_csv('ImageOutline.xls', sep = '\t', 
                                     engine='python')
    
    regional_start_positions(folder_with_tracks, image_outline_file)