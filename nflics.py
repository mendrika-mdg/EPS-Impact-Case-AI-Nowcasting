import regex as re
import os
import numpy as np
from netCDF4 import Dataset             # type: ignore
import netCDF4 as nc                    # type: ignore
from scipy import ndimage
from skimage import measure

def prepare_core(file, spatial_filter_size):
    if os.path.exists(file):        
        data = Dataset(file, "r")
        cores = data["msg_cores"][:]
        cores[np.isnan(cores)] = 0
        cores[cores > 0] = 1
        cores = ndimage.maximum_filter(cores, size=(spatial_filter_size,spatial_filter_size))
        return cores

def all_files_in(data_path):
    all_files = []
    for (dir_path, dir_names, file_names) in os.walk(data_path):
        for file in file_names:
            all_files.append(os.path.join(dir_path, file))
    return all_files

def format(time):
    if time[0] == "0":
        return int(time[1])
    else:
        return int(time)
    
def date_format(number):
    if number < 24:
        if len(str(number)) == 1:
            return f"0{number}"
        else:
            return str(number)
    else:
        return date_format(number - 24)

def move_time(time, step):
    return date_format(format(time) + step)

def get_time(filename):
    """
        This function get the date of an observation file
        input:spatial_filter_size
            filename: name of the file (type: str)
        output:
            time: {"year":, "month": month, "day": day, "hour": hour, "minute": minute} (type: dict)
    """    
    result = re.findall(r"([\d]+)\.nc$", filename)
    date_time = result[0]    
    year = date_time[0:4]
    month = date_time[4:6]
    day = date_time[6:8]
    hour = date_time[8:10]
    minute = date_time[10:]
    return {"year": year, "month": month, "day": day, "hour": hour, "minute": minute}


def get_date_as_path(filename):
    result = re.search(r"([\w\d\-/]+)Hist_cores_wa_([\d]+)\.nc$", filename)
    day_path = result[1]
    date = result[2]
    return f"{day_path}Hist_cores_wa_{date[:-4]}" 


def move_hour(time, hour_step):

    time["hour"] = move_time(time["hour"], hour_step)

    if time["hour"] >= "24":
        time["day"] = move_time(time["day"], 1)
        time["hour"] = move_time(time["hour"], -24)

    if (time["day"] == "31" and time["month"] in ["06","09"]):
        time["month"] = move_time(time["month"], 1)
        time["day"] = move_time(time["day"], -30)

    if time["day"] == "32" and time["month"] in ["07","08"]:
        time["month"] = move_time(time["month"], 1)
        time["day"] = move_time(time["day"], -31)
    
    year = time["year"]
    month = time["month"]
    day = time["day"]
    hour = time["hour" ]
    minute = time["minute"]

    file_path = f"{year}/{month}/{day}/Hist_cores_wa_{year}{month}{day}{hour}{minute}.nc"
    return time, file_path


def back_to_filename(filename):
    year = filename["year"]
    month = filename["month"]
    day = filename["day"]
    hour = filename["hour"]
    minute = filename["minute"]
    return f"Hist_cores_wa_{year}{month}{day}{hour}{minute}.nc"


def search(hour, minute, data_path):
    all_files = all_files_in(data_path)
    filtering_result = []
    for file in all_files:
        file_name = get_time(file)
        if file_name["minute"] == minute and file_name["hour"] == hour:
            filtering_result.append(file)
    return filtering_result

def search_ymd(year, month, day, data_path):    
    all_files = all_files_in(data_path)
    filtering_result = []
    for file in all_files:
        file_name = get_time(file)
        if file_name["day"] == day and file_name["month"] == month and file_name["year"] == year:
            filtering_result.append(file)
    return filtering_result


def compute_pc(dataset, spatial_filter_size, xlimit=1803):
    sum_cores = np.zeros((556,xlimit))
    number_of_days = len(dataset)
    for file in dataset:
        if os.path.exists(file):
            data = Dataset(file, "r")
            cores = data["msg_cores"][:,:xlimit]
            cores[np.isnan(cores)] = 0
            cores[cores > 0] = 1
            cores = ndimage.maximum_filter(cores, size=(spatial_filter_size,spatial_filter_size))
            sum_cores += cores
        else:
            continue        
    pc = sum_cores/number_of_days
    return pc


def core_index(core_value, msg_core):
    index_core = np.argwhere(msg_core == core_value) 
    y_core, x_core = index_core[0]
    return y_core, x_core


def source_area(y, x, filter_size=81):
    f = int(filter_size/2)
    Sy = (y-f, y+f+1)
    Sx = (x-f, x+f+1)
    return Sy, Sx

def test_identify_H0(hour, minute, data_path, Sy_min, Sy_max, Sx_min, Sx_max):

    dataset = search(hour, minute, data_path)
    H_0 = []
    
    for file in dataset:
        if os.path.exists(file):
            data = Dataset(file, "r")
        else:
            continue
        data = Dataset(file, "r")
        cores = data["msg_cores"][:]
                
        index = data["core_ind"][:]
        number_of_cores = len(index)
        
        cores[np.isnan(cores)] = 0
		# flatten the array and sort it
        
        flatened = np.array(cores).flatten()
        flatened.sort()

		# all x0 values
        list_of_core_values = list(set(flatened[-number_of_cores:]))
        list_of_core_values.sort()
        
        for core_value in list_of_core_values:
            x0_y, x0_x = core_index(core_value=core_value, msg_core=cores)
            if Sy_min <= x0_y <= Sy_max and Sx_min <= x0_x <= Sx_max:
                if file in H_0:
                    continue
                else:
                    H_0.append(file)    
    return H_0


def identify_H0(hour, minute, data_path, Sy_min, Sy_max, Sx_min, Sx_max):

    dataset = search(hour, minute, data_path)
    H_0 = []
    
    for file in dataset:
        if os.path.exists(file):
            data = Dataset(file, "r")
            lat = data["Pmax_lat"][:]
            lon = data["Pmax_lon"][:]

            for lt, ln in zip(lat[:], lon[:]):
                if X0(lt, ln):
                    y, x = X0(lt, ln)
                    if y and x and Sy_min <= y <= Sy_max and Sx_min <= x <= Sx_max:
                        if file in H_0:
                            continue
                        else:
                            H_0.append(file)
                else:
                    continue
        else:
            continue
    return H_0


def compute_pc_x0(hour, minute, H0_data, spatial_filter_size, xlimit):	
	dataset = [f"{date}{hour}{minute}.nc" for date in H0_data]
	pc_x0 = compute_pc(dataset=dataset, spatial_filter_size=spatial_filter_size, xlimit=xlimit)
	return pc_x0


def squareId_to_S0(square_id):
    result = re.search(r"([\d]+)\_([\d]+)\_([\d]+)\_([\d]+)", square_id)
    if result:
        Sy_min = int(result[1])
        Sy_max = int(result[3])
        Sx_min = int(result[2])
        Sx_max = int(result[4])
    return Sy_min, Sy_max, Sx_min, Sx_max


rectangle_data = nc.Dataset("/home/mendrika/mendrika-phd/codes/nflics/geoloc_grids/msg_rect_ALLhr_ninner41_wa.nc")
rect_id = rectangle_data["rect_id"][:]
def assign_S0_to(y, x):
    for square_id in rect_id:
        Sy_min, Sy_max, Sx_min, Sx_max = squareId_to_S0(square_id)
        if Sy_min <= y <= Sy_max and Sx_min <= x <= Sx_max:
            return Sy_min, Sy_max, Sx_min, Sx_max


geodata = nc.Dataset("/home/mendrika/mendrika-phd/codes/nflics/geoloc_grids/nxny1640_580_nxnyds164580_blobdx0.04491576_area4_n23_20_32.nc")
def X0(lat, lon):
    lats_mid = geodata["lats_mid"][:]
    lons_mid = geodata["lons_mid"][:]
    index_lat = np.abs(lat-lats_mid).round(1) <= 0.03 # within the dataset resolution
    index_lon = np.abs(lon-lons_mid).round(1) <= 0.03
    index_x0 =  index_lat * index_lon
    y, x = np.median(np.argwhere(index_x0 == True), axis=0)
    if not np.isnan(y) and not np.isnan(x):
        return int(y), int(x)
    

def alt_X0(input_latitude, input_longitude):
    # These values were determined empirically (comparing from nflics.X0 to search grid)
    min_latitude = 3.34   
    min_longitude = -23.91

    # Step size (original dataset resolution)
    step = 0.03

    # Calculate the index assuming a uniform grid spacing
    latitude_index = int(round((input_latitude - min_latitude) / step))
    longitude_index = int(round((input_longitude - min_longitude) / step))

    return latitude_index, longitude_index


def x0_from(lat, lon, raw_cores):
    x0 = []
    for lt, ln in zip(lat, lon):
        if X0(lt, ln):
            y, x = X0(lt, ln)
            core_val = raw_cores[y,x]
            if core_val is not  np.ma.masked:
                x0.append(raw_cores[y,x])            
    return x0

def top(rank, ls, reverse=True):
    if len(ls) >= 1:
        ls = list(set(ls))
        ls_copy = ls.copy()
        ls_copy.sort(reverse=reverse)
        return ls_copy[:rank]


def get_storm(binary_grid):
    
    # label each connected grid cell
    labels, number_of_groups = measure.label(np.array(binary_grid), connectivity=2, return_num=True)

    # taking the label of each identified group
    label_name = np.unique(labels)

    # removing the label 0 since that corresponds to the background
    core_label = label_name[label_name != 0]

    dict_storm_size = {}
    for label in core_label:
        num_cells = np.sum(labels==label)    
        dict_storm_size[str(label)] = num_cells * 9
    
    return {"labels": labels, "number_of_storms": number_of_groups, "size":dict_storm_size}



def get_x0_label(x0_power, cores, labels):    
    x0y, x0x = np.where(cores == x0_power)
    list_storm_label = []    
    for y, x in zip(x0y, x0x):
        storm_label = labels[y, x]
        list_storm_label.append(storm_label)
    return list_storm_label


def reliability_curve(y, y_pred, bin_size=0.1, count_pred_per_bin=100):
  y = np.array(y)
  y_pred = np.array(y_pred)
  bins = np.arange(0, 1+bin_size, bin_size)
  bin_centers = [(b1 + b2) / 2 for b1, b2 in zip(bins[:-1], bins[1:])]
  prop_positive = []
  bin_center_ax = []
  no_pred_per_bin = []
  for b1, b2, bc in zip(bins[:-1], bins[1:], bin_centers):
    in_bin = np.logical_and(y_pred >= b1, y_pred < b2)
    no_predictions_per_bin = np.sum(in_bin)
    no_positive_per_bin = np.sum(y[in_bin])    
    if no_predictions_per_bin > count_pred_per_bin:
      prop_pos = round(no_positive_per_bin/no_predictions_per_bin, 3)
      prop_positive.append(prop_pos)
      bin_center_ax.append(round(bc, 3))
      no_pred_per_bin.append(no_predictions_per_bin)
    else:
      continue
  return bin_center_ax, prop_positive, no_pred_per_bin


def downsample_grid(grid, factor):
    # Ensure that the grid dimensions are divisible by the factor
    assert grid.shape[0] % factor == 0
    assert grid.shape[1] % factor == 0

    # Reshape the grid to factor blocks and then take the mean
    grid_downsampled = grid.reshape(grid.shape[0] // factor, factor,
                                    grid.shape[1] // factor, factor).mean(axis=(1, 3))
    
    return grid_downsampled