import os
import re

def clean_filenames(path):
    """Standardizes Car and Bike filenames and converts them
    to .jpeg files"""
    bike_idx = 0
    car_idx = 0

    for filename in os.listdir(path):
        if filename.lower().startswith(r'bike'):
            org_filename = os.path.join(path + filename)
            new_filename = os.path.join(path + "bike_" + str(bike_idx)+ ".jpeg" )
            
            os.rename(org_filename, new_filename)
            bike_idx += 1

        elif filename.lower().startswith(r'car'):
            org_filename = os.path.join(path + filename)
            new_filename = os.path.join(path + "car_" + str(car_idx)+ ".jpeg" )
            
            os.rename(org_filename, new_filename)
            car_idx += 1

    print("file naming completed!")


