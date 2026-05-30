# This the basic code for remote sensing data preprocessing, where the stacked data is opened and prepared as feature
# predictors (x, x_test). the samples of ore deposit are processed as target variables (y, y_test)
# The code also contain how data is preprocessed to fit different ML models requirement.
# The code is mainly several function which can be called in others python codes within the same directory

from osgeo import gdal
from osgeo import ogr
import tensorflow as tf
import numpy as np
import geopandas as gpd
import os
import random
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math



driverTiff = gdal.GetDriverByName('GTiff')





# this function for opening the stacked data to extract the x_train and x_test
# as well and reshaping the image to become as np array
def rs_preprocessing (data, reshape=True):
    rs_ds = gdal.Open(data)
    nbands = rs_ds.RasterCount
    band_data = []
    print('bands', rs_ds.RasterCount, 'rows', rs_ds.RasterYSize, 'columns',
          rs_ds.RasterXSize)
    for i in range(1, nbands + 1):
        band = rs_ds.GetRasterBand(i).ReadAsArray()
        band_data.append(band)
    band_data = np.dstack(band_data)
    print(band_data.shape)

    if reshape == True:
        new_shape = (band_data.shape[0] * band_data.shape[1], band_data.shape[2])
        img_as_array = band_data[:, :, :np.int64(band_data.shape[2])].reshape(new_shape)
        print('Reshaped from {o} to {n}'.format(o=band_data.shape, n=img_as_array.shape))
        img_as_array = np.nan_to_num(img_as_array)
        return band_data, img_as_array
    else:
        return band_data





# the next function accept shapefile data contains points as target variables samples
# the target value must be in attribute called value and values are binary (0,1)
# 1 represents that the target mineral exist and vice versa
# the function splits the data into train and test datasets and save them separately in two files

def target_variable (data, trainDirectory, testDirectory, tarinPercent=0.8):
    gdf = gpd.read_file(data)
    # def condition(dataframe):
    #     if dataframe['value'] == 1:
    #         value = 3
    #     elif dataframe['value'] == 0.5:
    #         value = 2
    #     else:
    #         value = 1
    #     return value
    # gdf['raster'] = gdf.apply(condition, axis=1)
    # gdf['raster'] = np.where(gdf['value'] == 0, 1, 2)
    print(gdf.head)
    # gdf_train = gdf.sample(frac=tarinPercent)
    # gdf_test = gdf.drop(gdf_train.index)
    target_col = 'prospect'

    # Initialize StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - tarinPercent, random_state=42)

    # Perform stratified splitting
    for train_index, test_index in sss.split(gdf, gdf[target_col]):
        gdf_train = gdf.iloc[train_index]
        gdf_test = gdf.iloc[test_index]
    print('gdf shape', gdf.shape, 'training', gdf_train.shape, 'test', gdf_test.shape)
    gdf_train.to_file(trainDirectory)
    gdf_test.to_file(testDirectory)
    print('train data saved to: {}'.format(trainDirectory))
    print('test data saved to: {}'.format(testDirectory))


# Data rasterization and extraction of training and testing dataset
# The folowing function accept
# (1) The RS data directory
# (2) The processed RS data
# (3) the directory of the training or testing dataset
# function return x (variable features) and y (target variables)
def dataFitting (RSData, band_data, SHfile):
    RS_ds = gdal.Open(RSData)
    train_ds = ogr.Open(SHfile)
    lyr = train_ds.GetLayer()
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', RS_ds.RasterXSize, RS_ds.RasterYSize, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(RS_ds.GetGeoTransform())
    target_ds.SetProjection(RS_ds.GetProjectionRef())
    options = ['ATTRIBUTE=raster']
    gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
    data = target_ds.GetRasterBand(1).ReadAsArray()
    print('min', data.min(), 'max', data.max(), 'mean', data.mean())
    truth = target_ds.GetRasterBand(1).ReadAsArray()
    classes = np.unique(truth)[1:]
    print('class values', classes)
    n_samples = (data > 0).sum()
    print('{n} training samples'.format(n=n_samples))


    idx = np.nonzero(truth)
    x = band_data[idx]
    y = truth[idx] - 1

    # positive_label = 2
    # positive_mask = (truth == positive_label)
    # x = band_data[positive_mask]
    # y = truth[positive_mask] - 1


    
    
    # def condition(dataframe):
    #     if dataframe == 3:
    #         value = 1
    #     elif dataframe == 2:
    #         value = 0.5
    #     else:
    #         value = 0
    #     return value
    # y = list(map(condition, truth[idx]))
    print('Our X matrix is sized: {sz}'.format(sz=x.shape))
    print('Our y array is sized: {sz}'.format(sz=np.shape(y)))

    return x, y

def dataFitting_aug(RSData, band_data, SHfile, patch_size=3):
    print("band_data min:", np.min(band_data), "band_data max:", np.max(band_data))
    
    RS_ds = gdal.Open(RSData)
    train_ds = ogr.Open(SHfile)
    lyr = train_ds.GetLayer()
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', RS_ds.RasterXSize, RS_ds.RasterYSize, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(RS_ds.GetGeoTransform())
    target_ds.SetProjection(RS_ds.GetProjectionRef())
    options = ['ATTRIBUTE=raster']
    gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
    truth = target_ds.GetRasterBand(1).ReadAsArray()
    
    # Find indices where there are positive labels
    idx = np.nonzero(truth)
    print("Samples location", np.shape(idx))
    
    x = band_data[idx]  # Extract all the data at those indices
    y = truth[idx] - 1  # Adjust labels to be either 0 or 1 (assuming the original labels are 1 or 2)
    
    print(f'Our X matrix is sized: {x.shape}')
    print(f'Our y array is sized: {np.shape(y)}')
    
    # Extract patches around each sample position
    patches_ = []
    labels = []
    height, width, num_bands = band_data.shape  # Assuming band_data is (height, width, num_bands)
    half_patch = patch_size // 2
    
    for i in range(len(idx[0])):
        row, col = idx[0][i], idx[1][i]  # Get the position of the sample

        # Ensure that the patch is within bounds
        if row - half_patch >= 0 and row + half_patch < height and \
           col - half_patch >= 0 and col + half_patch < width:
            
            # Extract the patch centered around the sample
            patch = band_data[row - half_patch: row + half_patch + 1,
                              col - half_patch: col + half_patch + 1, :]
            patches_.append(patch)
            labels.append(y[i])  # Corresponding label (0 or 1)

            # Extract the patch centered around the sample
            patch = band_data[row - half_patch: row + half_patch + 1,
                              col - half_patch: col + half_patch + 1, :]
            
            patches_.append(patch)
            labels.append(y[i])  # Store only one label per patch
    
    # Convert patches to numpy array and reshape
    x_patches = np.array(patches_).reshape(-1, num_bands)  # Reshape to (990, 12)
    y_labels = np.repeat(np.array(labels), patch_size * patch_size)  # Repeat labels to match x_patches shape
    
    # Remove duplicate pixels and corresponding labels
    unique_x, unique_indices = np.unique(x_patches, axis=0, return_index=True)
    y_labels = y_labels[unique_indices]
    
    print(f'Patches shape after removing overlaps: {unique_x.shape}')
    print(f'Labels shape: {y_labels.shape}')

    p_idx = []
    n_idx = []

    for i in range (0, (y_labels.shape[0])):
        if y_labels[i] == 0:
            n_idx.append(i)
        else:
            p_idx.append(i)

    print("positive", len(p_idx))
    print("negative", len(n_idx))
    
    return unique_x, y_labels

def dataFitting_image(RSData, band_data, SHfile, patch_size=9):
    """
    Extract center-anchored image patches for Image-based MPM training.

    - Odd patch_size  -> true center pixel
    - Even patch_size -> top-left anchored center

    Parameters
    ----------
    RSData : str
        Reference raster path (for georeference)
    band_data : ndarray
        Feature raster of shape (H, W, C)
    SHfile : str
        Shapefile with training labels
    patch_size : int
        Patch size (odd or even)

    Returns
    -------
    x_patches : ndarray
        Extracted patches, shape (N, patch_size, patch_size, C)
    y_labels : ndarray
        Corresponding labels, shape (N,)
    """

    # ---------- determine patch mode ----------
    is_odd = (patch_size % 2 == 1)
    h = patch_size // 2

    # ---------- rasterize labels ----------
    RS_ds = gdal.Open(RSData)
    train_ds = ogr.Open(SHfile)
    lyr = train_ds.GetLayer()

    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create(
        '', RS_ds.RasterXSize, RS_ds.RasterYSize, 1, gdal.GDT_UInt16
    )
    target_ds.SetGeoTransform(RS_ds.GetGeoTransform())
    target_ds.SetProjection(RS_ds.GetProjectionRef())

    gdal.RasterizeLayer(target_ds, [1], lyr, options=['ATTRIBUTE=raster'])
    truth = target_ds.GetRasterBand(1).ReadAsArray()

    # ---------- prepare patch extraction ----------
    H, W, C = band_data.shape
    patches = []
    labels = []

    rows, cols = np.nonzero(truth)

    for row, col in zip(rows, cols):

        if is_odd:
            # ----- true center (odd patch) -----
            r0 = row - h
            r1 = row + h + 1
            c0 = col - h
            c1 = col + h + 1
        else:
            # ----- top-left anchored center (even patch) -----
            r0 = row - h
            r1 = row + h
            c0 = col - h
            c1 = col + h

        # ---------- safe border check ----------
        if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
            continue

        patch = band_data[r0:r1, c0:c1, :]
        patches.append(patch)

        # label always from center pixel
        labels.append(truth[row, col] - 1)

    x_patches = np.array(patches)
    y_labels = np.array(labels)

    print(f'Patch size: {patch_size} ({ "odd" if is_odd else "even, top-left anchored" })')
    print(f'Patches shape: {x_patches.shape}')
    print(f'Labels shape: {y_labels.shape}')
    print(f'Number of positive samples: {(y_labels == 1).sum()}')

    return x_patches, y_labels


    
    

def visualize_samples(band_data, patch_centers, patch_size=32, channel_num=0):
    """
    Visualize all patches on the first band of the study area.

    Args:
        band_data (numpy array): The entire raster data (height, width, num_bands).
        patches (numpy array): The extracted patches (num_patches, patch_size, patch_size, num_bands).
        patch_centers (list): The centers of the patches [(row, col), ...].
        patch_size (int): Size of the patches (default is 32x32).
    """
    # Extract the first band of the entire study area
    first_band = band_data[:, :, channel_num]

    # Plot the first band
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(first_band, cmap='RdYlBu_r', interpolation='none')
    ax.set_title("Patches Overlayed on First Band of Study Area")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Overlay each patch
    # for i, center in enumerate(patch_centers):
    #     row, col = center
    #     # Calculate the patch bounding box
    #     top_left = (col - patch_size // 2, row - patch_size // 2)
    #     # Create a rectangle patch
    #     rect = Rectangle(top_left, patch_size, patch_size, linewidth=1, edgecolor='red', facecolor='none')
    #     ax.add_patch(rect)

    #     # Optionally, show the patch inside the rectangle (scaled down for better clarity)
    #     patch = patches[i, :, :, 0]  # Use the first band of each patch
    #     ax.imshow(patch, extent=(top_left[0], top_left[0] + patch_size,
    #                              top_left[1] + patch_size, top_left[1]),
    #               cmap='jet', alpha=0.5)
        
    for center in patch_centers:
        row, col = center
        # Calculate the top-left corner of the bounding box
        top_left_row = row - patch_size // 2
        top_left_col = col - patch_size // 2

        # Add a rectangle to indicate the patch
        rect = Rectangle((top_left_col, top_left_row), patch_size, patch_size,
                         linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def visualize_first_patches(x_patches, num_patches=10, patches_per_row=5, channel_num=0):
    """
    Visualize the first `num_patches` patches using the first band.
    Starts a new row after `patches_per_row` patches.

    Args:
        x_patches (numpy array): The extracted patches (num_patches, patch_size, patch_size, num_bands).
        num_patches (int): Number of patches to display (default is 10).
        patches_per_row (int): Number of patches to display per row (default is 4).
    """
    # Ensure we don't exceed the number of available patches
    num_patches = min(num_patches, len(x_patches))
    
    # Calculate the number of rows needed
    num_rows = math.ceil(num_patches / patches_per_row)

    # Set up a grid for visualization
    fig, axes = plt.subplots(num_rows, patches_per_row, figsize=(patches_per_row * 4, num_rows * 4))
    fig.suptitle("Patches (First Band)", fontsize=16)

    # Flatten axes for easy indexing
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < num_patches:
            # Extract the first band of the patch
            patch_first_band = x_patches[i, :, :, channel_num]

            # Display the patch
            axes[i].imshow(patch_first_band, cmap='RdYlBu', interpolation='none')
            axes[i].set_title(f"Patch {i+1}")
            axes[i].axis('off')
        else:
            # Hide any extra subplots
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# def get_map_prediction()



def tfPipline (feature, label, shuffle=True, repeat=False, BUFFER_SIZE=10000, BATCH_SIZE=64):
    if label == '':
        dataset = tf.data.Dataset.from_tensor_slices(feature)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((feature, label))
    if repeat==True:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    else:
        dataset = dataset.batch(BATCH_SIZE)

    return dataset

def cnn_input(dataset):
    sample_size = dataset.shape[0]
    time_steps = dataset.shape[1]
    input_dimension = 1
    dataset = dataset.reshape(sample_size, time_steps, input_dimension)

    return dataset

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1)
   tf.random.set_seed(1)
   np.random.seed(1)
   random.seed(1)


def write_raster(RSData, modelPrediction, band_data, savedDirectory):
    RS_ds = gdal.Open(RSData)
    cols = band_data.shape[1]
    rows = band_data.shape[0]
    modelPrediction.astype(np.float16)
    driver = gdal.GetDriverByName("gtiff")
    outdata = driver.Create(savedDirectory, cols, rows, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(RS_ds.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(RS_ds.GetProjection())  ##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(modelPrediction)
    outdata.FlushCache()  ##saves to disk!!
    print('Image saved to: {}'.format(savedDirectory))

def main():
    print("This is the main code to test above functions")
    landsat = 'D:/programes/dataset/aster-finalstack2.tif'
    band_data1, img_as_array1 = rs_preprocessing(landsat, reshape=True)
    train_ds = 'D:/programes/qgis/train_reg.shp'
    x_train, y_train = dataFitting(landsat, band_data1, train_ds)
    print(x_train)
    print(y_train)

if __name__ == '__main__':
    main()

