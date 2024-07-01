#%%
import os
import json
from netCDF4 import Dataset

# Function to extract latitude, longitude, and area from a NetCDF file
def get_lat_lon_area(file_path):
    with Dataset(file_path, 'r') as nc_file:
        # Assuming the latitude and longitude variables are named 'lat' and 'lon'
        if 'latitude' in nc_file.variables and 'longitude' in nc_file.variables:
            latitudes = nc_file.variables['latitude'][:]
            longitudes = nc_file.variables['longitude'][:]
            
            # Extracting the 'comment' attribute for area information
            area = getattr(nc_file, 'comment', 'Unknown area')
            return latitudes, longitudes, area
    return None, None, None

directory_path = '../data/insitu/'  # Replace with your directory path
points = {}

# List NetCDF files in the directory
netcdf_files = [file for file in os.listdir(directory_path) if file.endswith('.nc')]

# Extract latitudes, longitudes, and area for each file and store in the dictionary
for file_name in netcdf_files:
    file_path = os.path.join(directory_path, file_name)
    latitudes, longitudes, area = get_lat_lon_area(file_path)
    
    if latitudes is not None and longitudes is not None and area is not None:
        # Assuming the file_name contains the desired key for the dictionary
        key = file_name.split('.')[0]  # Extracting the key from the file name
        
        # Saving data into the dictionary
        points[key] = {
            'x': float(longitudes),  # Assuming there's only one value for longitude and latitude
            'y': float(latitudes),
            'area': area
        }

# Print the points dictionary in the requested format
print("points = {")
for key, value in points.items():
    print(f'    "{key}": {{"x": {value["x"]}, "y": {value["y"]}, "area": \'{value["area"]}\' }},')
print("}")

with open('./pointsNDBC.info', 'w') as file:
    json.dump(points, file, indent=4)



# %%
