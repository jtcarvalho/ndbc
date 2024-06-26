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
from scipy.stats import linregress

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
            rmse = np.sqrt((np.sum(modelados[i] - observados[i])) ** 2)
            n_rmse = rmse / observados[i]
            n_rmse_percent = n_rmse * 100
            n_rmse_percent_values.append(n_rmse_percent)

    if len(n_rmse_percent_values) > 0:
        n_rmse_percent_mean = np.mean(n_rmse_percent_values)
    else:
        n_rmse_percent_mean = None

    return n_rmse_percent_mean

def plot_specific_buoys(ax, stations_info, prefix, color):
    specific_stations = [(float(stations_info[station]['Longitude']), float(stations_info[station]['Latitude'])) 
                         for station in stations_info if station.startswith(prefix) and stations_info[station] is not None]
    
    if specific_stations:
        lons, lats = zip(*specific_stations)
        ax.scatter(lons, lats, color=color, marker='o', s=10, transform=ccrs.PlateCarree())



def plot_individual_group_scatter(stations_info, prefix_groups, buoy_path, model_path, start_date, end_date, colors, group_labels):
    for idx, group in enumerate(prefix_groups):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel('Hs Buoy (m)', fontsize=12)
        ax.set_ylabel('Hs Model (m)', fontsize=12)
        ax.set_xlim([0, 8.])
        ax.set_ylim([0, 8.])

        #group_stations = [(station_id, stations_info[station_id]) for station_id in stations_info
        #                  if any(station_id.startswith(prefix) for prefix in group) and station_id in available_data]
        group_stations = [(station_id, stations_info[station_id]) for station_id in stations_info       
                        if any(station_id.startswith(prefix) for prefix in group)]
        
        if group_stations:
            hs_boias_valido_all = []
            hs_modelo_valido_all = []

            for station_id, station_data in group_stations:
                buoy_file = os.path.join(buoy_path, f"{station_id}.nc")
                #model_file = os.path.join(model_path, f"ww3_exp21_{station_id}.nc")
                model_file = os.path.join(model_path, f"ww3_{station_id}.nc")
                model_data = xr.open_dataset(model_file)
                buoy_data = xr.open_dataset(buoy_file)

                reference_date = pd.to_datetime('2020-01-01')  
                buoy_data['time'] = reference_date + pd.to_timedelta(buoy_data.time.values, unit='h')
                buoy_data['VHM0'] = xr.where(buoy_data['VHM0'] == 9.96921e+36, np.nan, buoy_data['VHM0'])


                buoy_data = buoy_data.sel(time=slice(start_date, end_date))
                model_data = model_data.sel(time=slice(start_date, end_date))

                #valid_dates_buoy = buoy_data['time'].where(~np.isnan(buoy_data['hs']), drop=True)
                valid_dates_buoy = buoy_data['time'].where(~np.isnan(buoy_data['VHM0']), drop=True)
                common_dates = pd.Index(valid_dates_buoy).intersection(model_data['time'])
                
                buoy_data = buoy_data.sel(time=common_dates)
                model_data = model_data.sel(time=common_dates)

                # hs_boias_valido = buoy_data['VHM0'].values
                # #hs_boias_valido = buoy_data['hs'].values
                # hs_modelo_valido = model_data['hs'].values

                valid_indices = np.where((buoy_data['VHM0'].notnull().values) & (model_data['hs'].values > 0.1))[0]
                hs_boias_valido = buoy_data['VHM0'].values[valid_indices]
                hs_modelo_valido = model_data['hs'].values[valid_indices]


                hs_boias_valido_all.extend(hs_boias_valido)
                hs_modelo_valido_all.extend(hs_modelo_valido)

                buoy_data.close()
                model_data.close()
   

            scatter = ax.scatter(hs_boias_valido_all, hs_modelo_valido_all, marker='o', s=8, color=colors[idx], alpha=0.45, label=group_labels[idx])

            # #Calcular média ou mediana para a linha de regressão

            x = np.array(hs_boias_valido_all)
            y = np.array(hs_modelo_valido_all)

            # x1 = np.linspace(0,len(x),len(x))
            slope1, intercept1, r, p, stderr = linregress(x, y)
            # min1=np.float64(np.min(hs_boias_valido))
            # max1=np.float64(np.max(hs_boias_valido))

            # ax.plot(x1,np.linspace(min1,max1,len(x1))*+slope1+intercept1,'b--',linewidth=2.0,label='regression line')
            # sns.regplot(data=buoy_data, x=x1, y=buoy_data, ax=ax, color='blue', x_estimator=0, scatter_kws={'s': 7}, line_kws={"lw":1.5, 'linestyle':'--'},scatter=False,label='regression line')
            if len(x) > 0 and len(y) > 0:
               m, b = np.polyfit(x, y, 1)
            else:
               m, b = 0, 0
            ax.plot(x, m * x + b, color='dimgrey', label='Real Fit')  # Linha de regressão com valores ajustados

            # #Linha de regressão perfeita
            # mean_x = np.mean(hs_boias_valido_all)
            # mean_y = np.mean(hs_modelo_valido_all)
            # perfect_m, perfect_b = np.polyfit([mean_x, 0], [mean_x, 0], 1)  # Ajustando a linha perfeita
            # ax.plot([0, 8],perfect_b, perfect_m * 8 + perfect_b, 'k--', label='Perfect Fit')
            line_x = [0, 8]
            line_y = [0, 8]
            ax.plot(line_x,line_y, 'k--', label='Perfect Fit')
        # Depois de plotar a linha de regressão, adicione o texto para r e stderr
            ax.text(7, 0.5, f'r = {r:.2f}', fontsize=10, ha='right', va='center')
            ax.text(7, 0.2, f'slope = {slope1:2.2f}', fontsize=10, ha='right', va='center')


        plt.title(f'Scatter plot for {group_labels[idx]}')
        plt.tight_layout()
        plt.legend(loc='upper left')
        save_name = f'../figs/scatter_{group_labels[idx].replace(" ", "_")}.png'  
        plt.savefig(save_name, dpi=300)
        plt.show()


def plotBuoysLoc(lons, lats, stations_info, set_global=True):
    fig = plt.figure(figsize=(10, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())

    if set_global:
        ax.set_global()

    land = cfeature.NaturalEarthFeature(
        category='physical', name='land', scale='110m',
        edgecolor='face', facecolor=(0.7, 0.7, 0.7)
    )
    ax.add_feature(land)
    ax.coastlines(resolution='10m')

    plot_specific_buoys(ax, stations_info, "41", "blue")
    plot_specific_buoys(ax, stations_info, "44", "blue")
    plot_specific_buoys(ax, stations_info, "42", "orange")
    plot_specific_buoys(ax, stations_info, "46", "magenta")
    plot_specific_buoys(ax, stations_info, "51", "red")
    plot_specific_buoys(ax, stations_info, "52", "red")
    plot_specific_buoys(ax, stations_info, "55", "darkviolet") 
    plot_specific_buoys(ax, stations_info, "56", "darkviolet") 
    plot_specific_buoys(ax, stations_info, "57", "darkviolet")  
    plot_specific_buoys(ax, stations_info, "60", "green")
    plot_specific_buoys(ax, stations_info, "61", "green")
    plot_specific_buoys(ax, stations_info, "62", "gold")
    plot_specific_buoys(ax, stations_info, "63", "gold")
    plot_specific_buoys(ax, stations_info, "13", "gold")
    plot_specific_buoys(ax, stations_info, "66", "gold")
    plot_specific_buoys(ax, stations_info, "67", "gold")   
    plot_specific_buoys(ax, stations_info, "70", "deepskyblue")
    plot_specific_buoys(ax, stations_info, "80", "chocolate")

    plt.title('Buoy Station Positions (NDBC and EMODNET)')
    plt.tight_layout()
    fig.savefig('../figs/buoysGroupsValidAll.jpeg', dpi=300)

def getStationInfoNDBC(buoy_path, arq):
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


def getStationInfoEMODNET(buoy_path, arq):
    file_path = os.path.join(buoy_path, arq)
    arq1 = xr.open_dataset(file_path)
    try:
        station_info = {
            'Latitude': arq1['latitude'][0].values,
            'Longitude': arq1['longitude'][0].values
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
    plt.title('Buoy Positions - NBIAS')
    plt.grid(True)
    plt.tight_layout()
    save_name = f'../figs/buoysAllNBIAS.jpeg'  
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
    plt.title('Buoy Positions - NRMSE')
    plt.grid(True)
    plt.tight_layout()
    save_name = f'../figs/buoysAllNRMSE.jpeg'  
    plt.savefig(save_name, dpi=300)



# NDBC buoys
# Definindo suas variáveis e caminhos
buoy_path1= '../data/insitu/'
model_path1 = '../data/ww3/points/'
#stations = ['42001', '42002']  

arqs_nc1 = [arq for arq in os.listdir(buoy_path1) if arq.endswith('.nc')]
stations = []
for arq in arqs_nc1:
    station_id = os.path.splitext(arq)[0]  # Get ID from file without extension
    stations.append(station_id) # List of stations IDs


all_latitudes1 = []
all_longitudes1 = []
n_bias_percent_values1 = []
n_rmse_percent_values1 = []

for station_id in stations:
    buoy_file1 = os.path.join(buoy_path1, f"{station_id}.nc")
    model_file1 = os.path.join(model_path1, f"ww3_{station_id}.nc")

    buoy_data1 = xr.open_dataset(buoy_file1)     
    model_data1= xr.open_dataset(model_file1)

    valid_dates_buoy1 = buoy_data1['time'].where(~np.isnan(buoy_data1['hs']), drop=True)
    common_dates1 = pd.Index(valid_dates_buoy1).intersection(model_data1['time'])
#     #common_dates = np.intersect1d(buoy_data['time'].values, model_data['time'].values)
    buoy_data_common1 = buoy_data1.sel(time=common_dates1)
    model_data_common1 = model_data1.sel(time=common_dates1)

    observed1 = buoy_data_common1['hs'].values
    modeled1 = model_data_common1['hs'].values
  
    n_bias_percent_value1 = calculate_nbias(observed1, modeled1)
    n_rmse_percent_value1 = calculate_nrmse(observed1, modeled1)

    latitudes1 = getStationInfoNDBC(buoy_path1, f"{station_id}.nc")['Latitude']
    longitudes1 = getStationInfoNDBC(buoy_path1, f"{station_id}.nc")['Longitude']

    print(f"Station {station_id}: NBIAS Percentage = {n_bias_percent_value1}")

    all_latitudes1.extend(latitudes1)
    all_longitudes1.extend(longitudes1)
    n_bias_percent_values1.append(n_bias_percent_value1)
    n_rmse_percent_values1.append(n_rmse_percent_value1)


# EMODNET buoys

# Definindo suas variáveis e caminhos
###buoy_path2 = '/work/opa/jc11022/projects/ww3GlobalUnst/data/buoy/emodnet/'
###model_path2 = '/work/opa/jc11022/projects/ww3GlobalUnst/v05_3km/work/output/points-emodnet/'

###arqs_nc2 = [arq for arq in os.listdir(buoy_path2) if arq.endswith('.nc')]
###stations = []
###for arq in arqs_nc2:
###    station_id = os.path.splitext(arq)[0]  # Get ID from file without extension
###    stations.append(station_id) # List of stations IDs

# stations = []
# for arq in arqs_nc2:
#     i_ini = arq.find('ww3_') + len('ww3_')
#     i_fim = arq.find('.nc')
#     station_id = arq[i_ini:i_fim]
#     stations.append(station_id) # List of stations IDs


# all_latitudes2 = []
# all_longitudes2 = []
# n_bias_percent_values2 = []
# n_rmse_percent_values2 = []

# for station_id in stations:

#     buoy_file2 = os.path.join(buoy_path2, f"{station_id}.nc")
#     model_file2 = os.path.join(model_path2, f"ww3_{station_id}.nc")

#     buoy_data2 = xr.open_dataset(buoy_file2)
#     model_data2 = xr.open_dataset(model_file2)

#     reference_date = pd.to_datetime('2020-01-01')  
#     buoy_data2['time'] = reference_date + pd.to_timedelta(buoy_data2.time.values, unit='h')
#     buoy_data2['VHM0'] = xr.where(buoy_data2['VHM0'] == 9.96921e+36, np.nan, buoy_data2['VHM0'])

#     valid_dates_buoy2 = buoy_data2['time'].where(~np.isnan(buoy_data2['VHM0']), drop=True)
#     common_dates2 = pd.Index(valid_dates_buoy2).intersection(model_data2['time'])
#     #common_dates = np.intersect1d(buoy_data['time'].values, model_data['time'].values)
#     buoy_data_common2 = buoy_data2.sel(time=common_dates2)
#     model_data_common2 = model_data2.sel(time=common_dates2)

#     observed2 = buoy_data_common2['VHM0'].values
#     modeled2= model_data_common2['hs'].values

#     n_bias_percent_value2 = calculate_nbias(observed2, modeled2)
#     n_rmse_percent_value2 = calculate_nrmse(observed2, modeled2)

#     latitudes2 = getStationInfo(buoy_path2, f"{station_id}.nc")['Latitude'][0]
#     longitudes2 = getStationInfo(buoy_path2, f"{station_id}.nc")['Longitude'][0]

#     print(f"Station {station_id}: NBIAS Percentage = {n_bias_percent_value2}")

#     all_latitudes2.append(latitudes2)
#     all_longitudes2.append(longitudes2)
#     n_bias_percent_values2.append(n_bias_percent_value2)
#     n_rmse_percent_values2.append(n_rmse_percent_value2)


# all_latitudes = all_latitudes1 + all_latitudes2
# all_longitudes = all_longitudes1 + all_longitudes2
# n_bias_percent_values = n_bias_percent_values1 + n_bias_percent_values2
# n_rmse_percent_values = n_rmse_percent_values1+ n_rmse_percent_values2

# start_date = pd.to_datetime('2020-01-01 00:00:00')
# end_date = pd.to_datetime('2020-12-31 23:00:00')
start_date = '2020-02-01 00:00:00'
end_date =  '2020-02-29 23:00:00'
#----get Stations info
stations_info = {}
for arq in arqs_nc1:
    station_id = os.path.splitext(arq)[0]  # Obtém o ID do arquivo sem a extensão
    stations_info[station_id] = getStationInfoNDBC(buoy_path1,arq)

lats1 = [float(stations_info[station]['Latitude']) for station in stations_info if stations_info[station] is not None]
lons1 = [float(stations_info[station]['Longitude']) for station in stations_info if stations_info[station] is not None]

###for arq in arqs_nc2:
###    station_id = os.path.splitext(arq)[0]  # Obtém o ID do arquivo sem a extensão
###    stations_info[station_id] = getStationInfoEMODNET(buoy_path2,arq)

###lats2 = [float(stations_info[station]['Latitude']) for station in stations_info if stations_info[station] is not None]
###lons2 = [float(stations_info[station]['Longitude']) for station in stations_info if stations_info[station] is not None]


#lons=lons1+lons2
#lats=lats1+lats2

lons=lons1
lats=lats1

# colors = ["seagreen", "royalblue",'orange','palevioletred','green']
# group_labels = ['Alaska and West U.S.', 'East U.S.','Central America','Hawaii and Pacific','Mediterraneo']
# prefix_groups = [['46'], ['41','44'],['42'],['51', '52'],['60','61']]

colors = ["green", "darkviolet",'gold','deepskyblue','chocolate']
group_labels = ['Mediterraneo', 'Australia', 'Iberia and N.Sea','Island','Chile']
prefix_groups = [['60','61'],['55','56','57'],['62','63','13','66','67'],['70'],['80']]



#plotBuoysLoc(lons,lats,stations_info)
#plot_individual_group_scatter(stations_info, prefix_groups, buoy_path1, model_path1, start_date, end_date, colors, group_labels)
plot_map_nbias(all_longitudes1,all_latitudes1,n_bias_percent_values1)
plot_map_nrmse(all_longitudes1,all_latitudes1,n_rmse_percent_values1)
# %%




# compareWW3close  ww3_56010.nc    ww3_6200041.nc  ww3_6200144.nc  ww3_6201013.nc  ww3_6201047.nc            ww3_Donostia-buoy.nc (6701))
# ww3_1300131.nc   ww3_56014.nc    ww3_6200042.nc  ww3_6200145.nc  ww3_6201015.nc  ww3_6201050.nc            ww3_Drangsnes.nc (7005)
# ww3_42055.nc     ww3_6100001.nc  ww3_6200082.nc  ww3_6200146.nc  ww3_6201017.nc  ww3_6201051.nc            ww3_Flateyjardufl.nc (7006)
# ww3_42067.nc     ww3_6100002.nc  ww3_6200083.nc  ww3_6200149.nc  ww3_6201018.nc  ww3_6201052.nc            ww3_Grimseyjarsund.nc (7007)
# ww3_42395.nc     ww3_6100196.nc  ww3_6200084.nc  ww3_6200165.nc  ww3_6201019.nc  ww3_6201053.nc            ww3_Hornafjardardufl.nc (7008)
# ww3_44072.nc     ww3_6100197.nc  ww3_6200085.nc  ww3_6200199.nc  ww3_6201024.nc  ww3_6201059.nc            ww3_Kogurdufl.nc (7009)
# ww3_46005.nc     ww3_6100280.nc  ww3_6200091.nc  ww3_6200200.nc  ww3_6201026.nc  ww3_6300110.nc            ww3_ooi-gs-gs01sumo_1.nc (8001)
# ww3_46259.nc     ww3_6100281.nc  ww3_6200092.nc  ww3_6201001.nc  ww3_6201027.nc  ww3_6300112.nc            ww3_Pasaia-station.nc (6702)
# ww3_55045.nc     ww3_6100417.nc  ww3_6200093.nc  ww3_6201003.nc  ww3_6201028.nc  ww3_66059.nc              ww3_Snorre-B.nc (6703)
# ww3_55052.nc     ww3_6100430.nc  ww3_6200094.nc  ww3_6201004.nc  ww3_6201029.nc  ww3_Blakksnes.nc  (7001)        ww3_Straumnesdufl.nc (7013)
# ww3_55053.nc     ww3_6200001.nc  ww3_6200127.nc  ww3_6201008.nc  ww3_6201045.nc  ww3_CapeBridgewater01.nc (57001)  ww3_Surtseyjardufl.nc (7014)
# ww3_56005.nc     ww3_6200024.nc  ww3_6200130.nc  ww3_6201009.nc  ww3_6201046.nc  ww3_CAPOMELE.nc (60001)
