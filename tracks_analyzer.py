# for timing
from datetime import datetime

# time how long the whole script takes to run after user input
total_script_start_time = datetime.now()

total_datetimestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")

from pathlib import Path
import pandas as pd
import sys
import os
import analysis_18
import regionalysis_1

anim_check = input(
    "Create animations? This adds a significant amount of time. ")
print("\n")

file_to_open = r'Spots in tracks statistics.xls'
second_file_to_open = r'Tracks Statistics.xls'
third_file_to_open = r'FirstOutline.xls'
fourth_file_to_open = r'FirstDorsalOutline.xls'
fifth_file_to_open = r'FirstIntermediateOutline.xls'
sixth_file_to_open = r'FirstVentralOutline.xls'
seventh_file_to_open = r'FirstAnteriorOutline.xls'
eighth_file_to_open = r'FirstPosteriorOutline.xls'
ninth_file_to_open = r'ImageOutline.xls'

try:
    paths_times = pd.read_excel('paths_times.xlsx')
except:
    print("Make sure paths_times.xlsx is present \
          in this folder and run the script again.")
    sys.exit("File not found")

for index, row in paths_times.iterrows():
    folder_with_tracks = row[r'path']
    path_to_open = Path(folder_with_tracks, file_to_open)
    second_path_to_open = Path(folder_with_tracks, second_file_to_open)
    third_path_to_open = Path(folder_with_tracks, third_file_to_open)
    fourth_path_to_open = Path(folder_with_tracks, fourth_file_to_open)
    fifth_path_to_open = Path(folder_with_tracks, fifth_file_to_open)
    sixth_path_to_open = Path(folder_with_tracks, sixth_file_to_open)
    seventh_path_to_open = Path(folder_with_tracks, seventh_file_to_open)
    eighth_path_to_open = Path(folder_with_tracks, eighth_file_to_open)
    ninth_path_to_open = Path(folder_with_tracks, ninth_file_to_open)
    spots_file = pd.read_csv(path_to_open, sep = '\t', engine='python')
    goodtracks_file = pd.read_csv(second_path_to_open, sep = '\t', 
                                  engine='python')
    outline_file = pd.read_csv(third_path_to_open, sep = '\t', 
                               engine='python')
    dorsal_outline_file = pd.read_csv(fourth_path_to_open, sep = '\t', 
                                      engine='python')
    intermediate_outline_file = pd.read_csv(fifth_path_to_open, sep = '\t', 
                                            engine='python')
    ventral_outline_file = pd.read_csv(sixth_path_to_open, sep = '\t', 
                                       engine='python')
    anterior_outline_file = pd.read_csv(seventh_path_to_open, sep = '\t', 
                                        engine='python')
    posterior_outline_file = pd.read_csv(eighth_path_to_open, sep = '\t', 
                                         engine='python')
    image_outline_file = pd.read_csv(ninth_path_to_open, sep = '\t', 
                                     engine='python')
    arch_results = 'WholeArch'
    dorsal_results = 'DorsalArch'
    intermediate_results = 'IntermediateArch'
    ventral_results = 'VentralArch'
    anterior_results = 'AnteriorArch'
    posterior_results = 'PosteriorArch'
    image_results = 'WholeImage'
    developmental_time = float(row['time'])
    
    analysis_18.tracks_analysis(
        folder_with_tracks, spots_file, goodtracks_file, 
        outline_file, image_outline_file, developmental_time, 
        arch_results, 'first arch', anim_check)
    analysis_18.tracks_analysis(
        folder_with_tracks, spots_file, goodtracks_file, 
        dorsal_outline_file, image_outline_file, developmental_time, 
        dorsal_results, 'dorsal first arch', anim_check)
    analysis_18.tracks_analysis(
        folder_with_tracks, spots_file, goodtracks_file, 
        intermediate_outline_file, image_outline_file, developmental_time, 
        intermediate_results, 'intermediate first arch', anim_check)
    analysis_18.tracks_analysis(
        folder_with_tracks, spots_file, goodtracks_file, 
        ventral_outline_file, image_outline_file, developmental_time, 
        ventral_results, 'ventral first arch', anim_check)
    analysis_18.tracks_analysis(
        folder_with_tracks, spots_file, goodtracks_file, 
        anterior_outline_file, image_outline_file, developmental_time, 
        anterior_results, 'anterior first arch', anim_check)
    analysis_18.tracks_analysis(
        folder_with_tracks, spots_file, goodtracks_file, 
        posterior_outline_file, image_outline_file, developmental_time, 
        posterior_results, 'posterior first arch', anim_check)
    analysis_18.tracks_analysis(
        folder_with_tracks, spots_file, goodtracks_file, 
        image_outline_file, image_outline_file, developmental_time, 
        image_results, 'whole image', anim_check)
    regionalysis_1.regional_start_positions(
        folder_with_tracks, image_outline_file)
    os.chdir('../')
    
print(f'Total run time: {datetime.now() - total_script_start_time}')
print("\n")