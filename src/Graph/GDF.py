import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
import pickle
from osgeo import gdal, ogr
import matplotlib.pyplot as plt
import geopandas as gpd

class MineralGraphBuilder:
    def __init__(self, raster_path, shapefile_path, window_size=3):
        self.raster_path = raster_path
        self.shapefile_path = shapefile_path
        self.window_size = window_size
        self.band_data = None
        self.flat_array = None
        self.labels = None
        self.graph = nx.Graph()

    def rs_preprocessing(self, data, reshape=True):
        """Load and reshape raster to (pixels, bands)"""
        rs_ds = gdal.Open(data)
        nbands = rs_ds.RasterCount
        band_data = [rs_ds.GetRasterBand(i).ReadAsArray() for i in range(1, nbands + 1)]
        band_data = np.dstack(band_data)
        print(f"Raster shape: {band_data.shape}")

        if reshape:
            flat_shape = (band_data.shape[0] * band_data.shape[1], band_data.shape[2])
            img_as_array = band_data.reshape(flat_shape)
            img_as_array = np.nan_to_num(img_as_array)
            print(f"Reshaped to: {img_as_array.shape}")
            return band_data, img_as_array
        return band_data

    def dataFitting(self, RSData, band_data, SHfile):
        """Rasterize shapefile labels"""
        RS_ds = gdal.Open(RSData)
        train_ds = ogr.Open(SHfile)
        lyr = train_ds.GetLayer()
        driver = gdal.GetDriverByName('MEM')
        target_ds = driver.Create('', RS_ds.RasterXSize, RS_ds.RasterYSize, 1, gdal.GDT_UInt16)
        target_ds.SetGeoTransform(RS_ds.GetGeoTransform())
        target_ds.SetProjection(RS_ds.GetProjectionRef())
        gdal.RasterizeLayer(target_ds, [1], lyr, options=['ATTRIBUTE=raster'])
        truth = target_ds.GetRasterBand(1).ReadAsArray()
        idx = np.nonzero(truth)
        x = band_data[idx]
        y = truth[idx] - 1

        height, width, num_bands = band_data.shape

        labeled_node_ids = []
        for i in range(len(idx[0])):
            row, col = idx[0][i], idx[1][i]
            flat_index = row * width + col
            labeled_node_ids.append(flat_index)

        print(f"Training samples: {x.shape[0]}")
        print(f"Label counts: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"First 10 labeled node IDs: {labeled_node_ids[:10]}")
        print(f'Number of labeled node IDs: {len(labeled_node_ids)}')

        return x, y, labeled_node_ids

    def dataFitting_aug(self, RSData, band_data, SHfile, patch_size=3):
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

        idx = np.nonzero(truth)
        y = truth[idx] - 1

        patches_, labels, labeled_node_ids = [], [], []
        index_map = []
        height, width, num_bands = band_data.shape
        half_patch = patch_size // 2

        for i in range(len(idx[0])):
            row, col = idx[0][i], idx[1][i]
            label = y[i]

            if row - half_patch >= 0 and row + half_patch < height and col - half_patch >= 0 and col + half_patch < width:
                patch = band_data[row - half_patch: row + half_patch + 1, col - half_patch: col + half_patch + 1, :]
                patches_.append(patch)
                labels.append(label)

                for r in range(row - half_patch, row + half_patch + 1):
                    for c in range(col - half_patch, col + half_patch + 1):
                        flat_idx = r * width + c
                        labeled_node_ids.append(flat_idx)


        # x_patches = np.array(patches_).reshape(-1, num_bands)
        # y_labels = np.repeat(np.array(labels), patch_size * patch_size)
        # node_ids = np.repeat(np.array(index_map), patch_size * patch_size)
        x_patches = np.concatenate([p.reshape(-1, num_bands) for p in patches_], axis=0)
        y_labels = np.concatenate([[label] * (patch_size * patch_size) for label in labels])
        labeled_node_ids = np.array(labeled_node_ids)


        # Remove overlaps
        unique_x, unique_indices = np.unique(x_patches, axis=0, return_index=True)
        y_labels = y_labels[unique_indices]
        labeled_node_ids = labeled_node_ids[unique_indices]

        print(f'Patches shape after removing overlaps: {unique_x.shape}')
        print(f'Labels shape: {y_labels.shape}')
        print(f'Label counts: {dict(zip(*np.unique(y_labels, return_counts=True)))}')
        print(f'Number of labeled node IDs: {len(labeled_node_ids)}')
        print(f"First 10 labeled node indices: {labeled_node_ids[:10]}")

        return unique_x, y_labels, labeled_node_ids



    def load_and_preprocess(self, aug=False):
        self.band_data, self.flat_array = self.rs_preprocessing(self.raster_path)

        if aug:
            x, y, labeled_node_ids = self.dataFitting_aug(self.raster_path, 
                                                          self.band_data, 
                                                          self.shapefile_path)
        else:
            x, y, labeled_node_ids = self.dataFitting(self.raster_path, 
                                                      self.band_data, 
                                                      self.shapefile_path)

        total_nodes = self.band_data.shape[0] * self.band_data.shape[1]
        labels_array = np.full((total_nodes,), -1)
        labels_array[labeled_node_ids] = y[:len(labeled_node_ids)]  # Truncate in case of shape mismatch
        
        # Debug: count labels
        unique, counts = np.unique(labels_array, return_counts=True)
        label_distribution = dict(zip(unique, counts))
        print("Label distribution in labels_array:", label_distribution)


        self.labels = labels_array
        return self

    def _get_node_type(self, row, col, max_row, max_col):
        if (row < 1 or row >= max_row - 1 or col < 1 or col >= max_col - 1):
            return 'corner' if (row in [0, max_row - 1] and col in [0, max_col - 1]) else 'edge'
        return 'center'

    def build_connected_graph(self):
        self.graph = nx.Graph()
        rows, cols = self.band_data.shape[:2]
        total_nodes = rows * cols

        if self.labels is None or len(self.labels) != total_nodes:
            raise ValueError("Label size mismatch or not loaded.")

        flat_array = self.band_data.reshape((-1, self.band_data.shape[2]))

        for idx in range(total_nodes):
            row, col = divmod(idx, cols)
            node_type = self._get_node_type(row, col, rows, cols)
            label = self.labels[idx]
            self.graph.add_node(idx, features=flat_array[idx], label=label, position=(row, col), node_type=node_type)

        for idx in range(total_nodes):
            row, col = divmod(idx, cols)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        n_idx = nr * cols + nc
                        if not self.graph.has_edge(idx, n_idx):
                            weight = 1.0 / np.sqrt(dr**2 + dc**2)
                            self.graph.add_edge(idx, n_idx, weight=weight)

        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self


    @staticmethod
    def is_border_node(row, col, rows, cols):
        """Returns True if node is at image border."""
        return row == 0 or row == rows-1 or col == 0 or col == cols-1

    @staticmethod
    def ordered_neighbor_features(row, col, rows, cols, graph, feature_dim):
        """
        Generates ordered 9-slot vector [TL,T,TR,L,C,R,BL,B,BR].
        Pads missing slots with zeros if outside image.
        """
        slots = []
        for i, j in [(-1,-1), (-1,0), (-1,1),
                     (0,-1),           (0,1),
                     (1,-1),  (1,0),  (1,1)]:
            nr, nc = row + i, col + j
            if 0 <= nr < rows and 0 <= nc < cols:
                idx = nr * cols + nc
                slots.append(graph.nodes[idx]['features'])
            else:
                slots.append(np.zeros(feature_dim))
        return np.concatenate(slots, axis=0)

    def prepare_rf_training_data_with_heads(self, n_trees=100, min_samples=2, alpha = 0.2,
                                             random_state=None, num_heads=3):
        """Trains RFs for each head on labeled nodes, updates features with message passing."""
        rows, cols, bands = self.band_data.shape
        feature_dim = self.flat_array.shape[1]
        rf_models = []
        head_feature_matrices = []

        for head in range(num_heads):
            X_train, y_train = [], []

            head_initial_features = {
                n: self.graph.nodes[n]['features'].copy()
                for n in self.graph.nodes
            }


            for node in self.graph.nodes:
                node_data = self.graph.nodes[node]
                if node_data['label'] != 1 and node_data['label'] != 0:
                    continue
                row, col = node_data['position']

                # ordered or unordered feature vector
                if self.is_border_node(row, col, rows, cols):
                    vector = self.ordered_neighbor_features(row, col, rows, cols, self.graph, feature_dim)
                else:
                    neighbor_features = [self.graph.nodes[n]['features'] for n in sorted(self.graph.neighbors(node))]
                    vector = np.concatenate(neighbor_features, axis=0)

                X_train.append(vector)
                y_train.append(node_data['features'])

            X_train, y_train = np.array(X_train), np.array(y_train)

            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            # print("Sample y_train values:", y_train[:2])


            if X_train.ndim < 2 or X_train.shape[0] == 0:
                print(f"[Head {head}] No training samples found.")
                continue
            else:
                print(f"[Head {head}] Training RF on {X_train.shape[0]} samples of dim {X_train.shape[1]}")


            print()
            print("----------------")
            # print("first sample features:", X_train[0])
            # print("second sample features:", X_train[1])
            print(f"[Head {head}] Training RF on {X_train.shape[0]} samples of dim {X_train.shape[1]}")

            rf = RandomForestRegressor(n_estimators=n_trees,
                                       min_samples_split=min_samples,
                                       random_state=random_state, n_jobs=1)
            rf.fit(X_train, y_train)
            rf_models.append(rf)

            print(f"[Head {head}] Preparing data for message passing on {len(self.graph.nodes)} nodes...")

            vectors = []
            node_list = []

            for node in self.graph.nodes:
                node_data = self.graph.nodes[node]
                row, col = node_data['position']
                if self.is_border_node(row, col, rows, cols):
                    vector = self.ordered_neighbor_features(row, col, rows, cols, self.graph, feature_dim)
                else:
                    neighbor_features = [self.graph.nodes[n]['features'] for n in sorted(self.graph.neighbors(node))]
                    vector = np.concatenate(neighbor_features, axis=0)
                vectors.append(vector)
                node_list.append(node)


            batch_size = 10000  # You can adjust based on your RAM
            num_batches = (len(node_list) + batch_size - 1) // batch_size  # total number of batches

            print(f"[Head {head}] Preparing data for message passing on {len(node_list)} nodes with {num_batches} batches...")

            for batch_idx, start in enumerate(range(0, len(node_list), batch_size)):
                end = min(start + batch_size, len(node_list))
                batch_nodes = node_list[start:end]
                batch_vectors = np.stack([vectors[i].astype(np.float32) for i in range(start, end)], axis=0)

                batch_preds = rf.predict(batch_vectors)

                for idx, node in enumerate(batch_nodes):
                    # self.graph.nodes[node]['features'] = batch_preds[idx]
                    # alpha = 0.2  # or tune

                    h0 = head_initial_features[node]
                    h_rf = batch_preds[idx]

                    self.graph.nodes[node]['features'] = (
                        (1 - alpha) * h0 + alpha * h_rf
                    ) # the upated feature is PPR result


                print(f"[Head {head}] Batch {batch_idx + 1}/{num_batches} processed ({end}/{len(node_list)} nodes).")

            print(f"[Head {head}] Features updated on entire graph.")

            features = np.vstack([
                self.graph.nodes[n]['features'] for n in sorted(self.graph.nodes)
            ])
            head_feature_matrices.append(features.copy())  # Safe copy for external usage



            # Optionally save model for reuse
            with open(f'rf_head_{head}.pkl', 'wb') as f:
                pickle.dump(rf, f)

        # This tracks the same labeled node indices across heads
        labeled_node_ids = []
        y_labeled = []
        position = []

        for node in sorted(self.graph.nodes):
            node_data = self.graph.nodes[node]
            if 'label' in node_data and node_data['label'] in [0, 1]:
                labeled_node_ids.append(node)
                y_labeled.append(node_data['label'])
            if 'position' in node_data:
                position.append(node_data['position'])

        print()
        print("---------------")
        print('shape of the head feature matrices:', np.shape(head_feature_matrices))
        print('features of the first labled node:', head_feature_matrices[0][labeled_node_ids[0]])

        print("Training and message passing complete.")

    
            
        
         # Now extract labeled features per head
        print("Collecting labeled features after message passing...")
        X_labeled_per_head = []

        for head_idx, head_feats in enumerate(head_feature_matrices):
            labeled_features = []
            for node in labeled_node_ids:
                node_idx = list(self.graph.nodes).index(node)
                labeled_features.append(head_feats[node_idx])
            X_labeled_per_head.append(np.array(labeled_features))

        print("Done.")

        return head_feature_matrices, rf_models, X_labeled_per_head, np.array(y_labeled), position


    def get_test_data_from_shapefile(self, head_feature_matrices, test_shapefile_path):
        

        # Load test shapefile and extract coordinates and labels
        gdf = gpd.read_file(test_shapefile_path)
        coords = [(geom.xy[0][0], geom.xy[1][0]) for geom in gdf.geometry]
        y_test = gdf['raster'].values - 1  # Or change if your column is named differently

        # Get raster geotransform info
        ds = gdal.Open(self.raster_path)
        gt = ds.GetGeoTransform()
        width = ds.RasterXSize
        height = ds.RasterYSize

        # Map world coords → pixel row/col
        def world_to_pixel(x, y):
            col = int((x - gt[0]) / gt[1])
            row = int((y - gt[3]) / gt[5])
            return row, col

        test_node_ids = []
        for x, y in coords:
            row, col = world_to_pixel(x, y)
            if 0 <= row < height and 0 <= col < width:
                node_idx = row * width + col
                test_node_ids.append(node_idx)

        test_node_ids = np.array(test_node_ids)

        print("test node IDs:", test_node_ids[:20])


        X_labeled_per_head = []

        for head_idx, head_feats in enumerate(head_feature_matrices):
            labeled_features = []
            for node in test_node_ids:
                node_idx = list(self.graph.nodes).index(node)
                labeled_features.append(head_feats[node_idx])
            X_labeled_per_head.append(np.array(labeled_features))

        print("test features and labels extraction is done.")


        return X_labeled_per_head, y_test


    
    def save_all_features_map(self, save_path):
        """
        Saves the entire feature vector of each node as a multi-band raster GeoTIFF,
        where each band corresponds to one feature dimension.
        """
        rows, cols, _ = self.band_data.shape
        feature_dim = self.flat_array.shape[1]  # Number of features per node
        
        # Prepare an empty 3D array: (bands, rows, cols)
        feature_cube = np.zeros((feature_dim, rows, cols), dtype=np.float32)

        for idx in self.graph.nodes:
            row, col = self.graph.nodes[idx]['position']
            features = self.graph.nodes[idx]['features']
            feature_cube[:, row, col] = features  # Note the band-first order

        RS_ds = gdal.Open(self.raster_path)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(
            save_path, cols, rows, feature_dim, gdal.GDT_Float32
        )
        outdata.SetGeoTransform(RS_ds.GetGeoTransform())
        outdata.SetProjection(RS_ds.GetProjection())

        for b in range(feature_dim):
            outdata.GetRasterBand(b + 1).WriteArray(feature_cube[b])

        outdata.FlushCache()
        print(f"Multi-band feature map saved to: {save_path}")


# Updated usage example
if __name__ == "__main__":
    processor = MineralGraphBuilder(
        raster_path='E:/PhD/Data/Block 23/features/features.tiff',
        shapefile_path='E:/PhD/Data/Block 23/features/training70.shp'
    )

    # Process data
    processor.load_and_preprocess()
    
    # Build FULL connected graph (now with proper subgraph connections)
    processor.build_connected_graph()
    
    # Verify connectivity
    print(f"Graph has {processor.graph.number_of_nodes()} nodes")
    print(f"Graph has {processor.graph.number_of_edges()} edges")
    # print(f"Is graph connected? {nx.is_connected(processor.graph)}")
    

    head_feature_matrices, rf_models, X_labeled_per_head, y_labels = processor.prepare_rf_training_data_with_heads(num_heads=2)

    print(np.shape(head_feature_matrices))
    print(np.shape(X_labeled_per_head))
    print(np.shape(y_labels))
    print(y_labels[0])
    
    # save_dir = "E:/PhD/Data/Block 23/Graph/2heads.tiff"
    # processor.save_all_features_map(save_dir)
