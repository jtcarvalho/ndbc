#%%
import os
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

def calculate_nbias(observados, modelados):
    n_bias_percent_values = []

    for i in range(len(observados)):
        if not np.isnan(observados[i]) and not np.isnan(modelados[i]):
            bias = np.sum(modelados[i] - observados[i])
            n_bias = bias / observados[i]
            n_bias_percent = n_bias * 100
            n_bias_percent_values.append(n_bias_percent)

    if len(n_bias_percent_values) > 0:
        n_bias_percent_mean = np.mean(n_bias_percent_values)
    else:
        n_bias_percent_mean = None

    return n_bias_percent_mean


def calculate_nrmse(observados, modelados):
    n_rmse_percent_values = []

    for i in range(len(observados)):
        if not np.isnan(observados[i]) and not np.isnan(modelados[i]):
            rmse = np.sqrt((np.sum(modeled[i] - observed[i])) ** 2)
            n_rmse = rmse / observados[i]
            n_rmse_percent = n_rmse * 100
            n_rmse_percent_values.append(n_rmse_percent)

    if len(n_rmse_percent_values) > 0:
        n_rmse_percent_mean = np.mean(n_rmse_percent_values)
    else:
        n_rmse_percent_mean = None

    return n_rmse_percent_mean



def getStationInfo(buoy_path, arq):
    file_path = os.path.join(buoy_path, arq)
    arq1 = xr.open_dataset(file_path)
    try:
        station_info = {
            'Latitude': arq1['latitude'].values,
            'Longitude': arq1['longitude'].values
        }
    except Exception as e:
        print(f"Failed to fetch information for file: {arq}. Error: {e}")
        station_info = None
    finally:
        arq1.close()

    return station_info

def create_custom_cmap(colors, n_colors):
    # Create a color map with the defined colors and number of intervals
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_colors)
    return custom_cmap



def plot_map_nbias(all_longitudes, all_latitudes, n_bias_percent_values):
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)

    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-80, 81, 20))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.top_labels = False
    gl.right_labels = False

    # Resto do código para plotar o mapa...
    n_bias_percent_mean = np.nanmean([value for value in n_bias_percent_values if np.isfinite(value)])
    lon_text = 58  
    lat_text = 50   
    ax.text(lon_text, lat_text, r"$\overline{\mathrm{NBIAS}}=$"+f"{n_bias_percent_mean:.2f}%", color='k', transform=ccrs.PlateCarree(), fontsize=12)

    colors1 = ['deepskyblue', 'lightgreen', 'khaki', 'orange', 'red']
    n_colors = 32
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors1, N=n_colors)
    norm = mcolors.TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    sc = plt.scatter(all_longitudes, all_latitudes, c=n_bias_percent_values, cmap=custom_cmap, s=50, alpha=0.7, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('NBIAS (%)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('NDBC Buoy Positions - NBIAS')
    plt.grid(True)
    plt.tight_layout()
    save_name = f'../figs/ndbcBuoysNBIAS.jpeg'  
    plt.savefig(save_name, dpi=300)


def plot_map_nrmse(all_longitudes, all_latitudes, n_rmse_percent_values):
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)

    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-80, 81, 20))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.top_labels = False
    gl.right_labels = False

    # Resto do código para plotar o mapa...
    n_rmse_percent_mean = np.nanmean([value for value in n_rmse_percent_values if np.isfinite(value)])
    lon_text = 58  
    lat_text = 50   
    ax.text(lon_text, lat_text, r"$\overline{\mathrm{NRMSE}}=$"+f"{n_rmse_percent_mean:.2f}%", color='k', transform=ccrs.PlateCarree(), fontsize=12)

    colors1 = ['deepskyblue', 'lightgreen', 'khaki', 'orange', 'red']
    n_colors = 32
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors1, N=n_colors)
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=30, vmax=60)
    sc = plt.scatter(all_longitudes, all_latitudes, c=n_rmse_percent_values, cmap=custom_cmap, s=50, alpha=0.7, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('NRMSE (%)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('NDBC Buoy Positions - NRMSE')
    plt.grid(True)
    plt.tight_layout()
    save_name = f'../figs/ndbcBuoysNRMSE.jpeg'  
    plt.savefig(save_name, dpi=300)




# Definindo suas variáveis e caminhos
buoy_path = '../data/insitu/'
model_path = '../data/ww3/points/'
#stations = ['42001', '42002']  

arqs_nc = [arq for arq in os.listdir(buoy_path) if arq.endswith('.nc')]
stations = []
for arq in arqs_nc:
    station_id = os.path.splitext(arq)[0]  # Get ID from file without extension
    stations.append(station_id) # List of stations IDs


all_latitudes = []
all_longitudes = []
n_bias_percent_values = []
n_rmse_percent_values = []

for station_id in stations:
    buoy_file = os.path.join(buoy_path, f"{station_id}.nc")
    model_file = os.path.join(model_path, f"ww3_{station_id}.nc")

    buoy_data = xr.open_dataset(buoy_file)
    model_data = xr.open_dataset(model_file)

    valid_dates_buoy = buoy_data['time'].where(~np.isnan(buoy_data['hs']), drop=True)
    common_dates = pd.Index(valid_dates_buoy).intersection(model_data['time'])
    #common_dates = np.intersect1d(buoy_data['time'].values, model_data['time'].values)
    buoy_data_common = buoy_data.sel(time=common_dates)
    model_data_common = model_data.sel(time=common_dates)

    observed = buoy_data_common['hs'].values
    modeled = model_data_common['hs'].values

    n_bias_percent_value = calculate_nbias(observed, modeled)
    n_rmse_percent_value = calculate_nrmse(observed, modeled)

    latitudes = getStationInfo(buoy_path, f"{station_id}.nc")['Latitude']
    longitudes = getStationInfo(buoy_path, f"{station_id}.nc")['Longitude']

    print(f"Station {station_id}: NBIAS Percentage = {n_bias_percent_value}")

    all_latitudes.extend(latitudes)
    all_longitudes.extend(longitudes)
    n_bias_percent_values.append(n_bias_percent_value)
    n_rmse_percent_values.append(n_rmse_percent_value)

plot_map_nbias(all_longitudes,all_latitudes,n_bias_percent_values)
plot_map_nrmse(all_longitudes,all_latitudes,n_rmse_percent_values)
# %%
