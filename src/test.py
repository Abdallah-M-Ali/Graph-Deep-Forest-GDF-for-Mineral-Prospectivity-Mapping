from Data_preprocessing import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import numpy as np
import time as t
from Graph.DF import DFCascadeRegressor
from Graph.GDF import MineralGraphBuilder


features = 'features.tiff'
trainDirectory = 'training70.shp'
testDirectory = 'testing30.shp'

band_data_, img_as_array = rs_preprocessing(features, reshape=True)

n_estimators= [2, 4, 8, 16]
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_trees= [100, 200, 300, 400, 500]


# empty list of the accuracy metrics.
param_1 = []
param_2 = []
param_3 = []
MSE = []
roc_auc = []
OA = []


for j in n_trees:
    for k in alpha:

        processor = MineralGraphBuilder(
        raster_path=features,
        shapefile_path=trainDirectory
            )

            # Process data
        processor.load_and_preprocess()

        # Build FULL connected graph (now with proper subgraph connections)
        processor.build_connected_graph()

        print('-------------------------------------')

        head_feature_matrices, rf_models, X_labeled_per_head, y_train, full_position = processor.prepare_rf_training_data_with_heads(n_trees=j,
                                                                                                                                    min_samples=2,
                                                                                                                                    alpha = k,
                                                                                                                                    random_state=19,
                                                                                                                                    num_heads=3)
        
        # if you want to save the feature importance values for all heads and all layers, you can uncomment the following lines and save them in a .npy file for later analysis.
        
        # all_heads_importance = []

        # for h, rf in enumerate(rf_models):
        #     importance = rf.feature_importances_
            
        #     all_heads_importance.append(importance)
        
        # all_heads_importance = np.array(all_heads_importance)
        # print(f'feature importance shape: {all_heads_importance.shape}')
        # np.save('importance.npy', all_heads_importance)

        x_train = [X_labeled_per_head[i] for i in range(len(X_labeled_per_head))]


        X_test_heads, y_test = processor.get_test_data_from_shapefile(head_feature_matrices, testDirectory)
        x_test = [X_test_heads[i] for i in range(len(X_test_heads))]

        for i in n_estimators:

            # print('\nmodel start training for n_estimators:', i)

            Df = DFCascadeRegressor(n_estimators=i, n_trees=j,
                                    max_depth=None, min_samples_split=2,
                                    n_tolerant_rounds=5, random_state=42,
                                    validation_mode='oob', use_predictor=True)


            param_1.append(i)
            param_2.append(k)
            param_3.append(j)

            # start_time = t.time()
            Df.fit(x_train, y_train)
            # end_time = t.time()
            y_predicted = Df.predict(x_test)
            y_predicted = y_predicted.flatten()
            auc = roc_auc_score(y_test, y_predicted)
            roc_auc.append(auc)
            mse = mean_squared_error(y_test, y_predicted)
            MSE.append(mse)
            round_prediction = [round(l) for l in y_predicted]
            OA_accuracy = accuracy_score(y_test, round_prediction)
            OA.append(OA_accuracy)

            print('MSE:', mse, 'AUC:', auc, 'OA:', OA_accuracy)
            # Df.reset() # reset the model for the next iteration while parameters configuration is changing
            feature_importance = Df.feature_importances_()
            for layer_idx, imp_array in feature_importance.items():
                print(f'layer {layer_idx} feature importance shape: {imp_array.shape}')
                print(f"Layer {layer_idx} feature importatnce values:\n{imp_array}")

                layer_mean = np.mean(imp_array, axis=0)  # shape (12,)
                print(f"Layer {layer_idx} average feature importance:\n{layer_mean}")
            print('.................')


Data = {
        'n_trees':param_3,
        'alpha':param_2,
        'estimators':param_1,
        'MSE':MSE,
        'AUC':roc_auc,
        'OA':OA
        }

DataF = pd.DataFrame(Data)
DataF = DataF.sort_values(['MSE', 'AUC'], ascending=[True, False])
print(DataF)
# DataF.to_excel('training.xlsx')


# if you want to save the feature importance values for all heads and all layers, you can uncomment the following lines and save them in a .npy file for later analysis.

# final_stage = [head_feature_matrices[i] for i in range(len(head_feature_matrices))]
# print("----------------")
# print(f"full data shape: {np.shape(final_stage)}")


# MPM_predicted = Df.predict(final_stage).flatten()


# def save_raster(RSData, modelPrediction, band_data, position, savedDirectory):
#     RS_ds = gdal.Open(RSData)
#     cols = band_data.shape[1]
#     rows = band_data.shape[0]
    
#     pred = modelPrediction.astype(np.float32)
    
#     # Create an empty 2D array for the prediction map
#     prediction_map = np.zeros((rows, cols), dtype=np.float32)

#     # Fill the prediction map at the correct positions
#     for idx in range(len(position)):
#         row, col = position[idx]
#         prediction_map[row, col] = pred[idx]

#     # Write to a single-band raster
#     driver = gdal.GetDriverByName("GTiff")
#     outdata = driver.Create(savedDirectory, cols, rows, 1, gdal.GDT_Float32)
#     outdata.SetGeoTransform(RS_ds.GetGeoTransform())
#     outdata.SetProjection(RS_ds.GetProjection())
#     outdata.GetRasterBand(1).WriteArray(prediction_map)
#     outdata.FlushCache()

#     print(f"Prediction map saved to: {savedDirectory}")



# save_dir = "GDF.tiff"
# save_raster(features, MPM_predicted, band_data_, full_position, save_dir)
