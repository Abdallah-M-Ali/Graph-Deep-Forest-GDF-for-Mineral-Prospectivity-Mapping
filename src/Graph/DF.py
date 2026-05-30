from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_predict
import numpy as np
from sklearn.metrics import mean_squared_error
import gc


class DFCascadeRegressor:
    def __init__(self,
                 max_layers=20,
                 n_estimators=2,
                 n_trees=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 criterion='squared_error',
                 n_tolerant_rounds=2,
                 delta=1e-5,
                 n_jobs=-1,
                 random_state=None,
                 verbose=1,
                 validation_mode='oob',
                 n_folds=5,
                 use_predictor=False,
                 predictor_name="rf"):

        self.max_layers = max_layers
        self.n_estimators = n_estimators
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.n_tolerant_rounds = n_tolerant_rounds
        self.delta = delta
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.validation_mode = validation_mode
        self.n_folds = n_folds
        self.use_predictor = use_predictor
        self.predictor_name = predictor_name.lower()
        self.final_predictor = None

        self.models = []
        self.oob_scores = []


    def oob_decision_function_(self, estimator_):
        # Scikit-Learn uses `oob_prediction_` for ForestRegressor
        oob_prediction = estimator_.oob_prediction_
        if len(oob_prediction.shape) == 1:
            oob_prediction = np.expand_dims(oob_prediction, 1)
        return oob_prediction
    
    def fit_transform(self, X, y, estimator_):
        estimator_.fit(X, y)
        # return self.oob_decision_function_(estimator_)
        pred = self.oob_decision_function_(estimator_)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    def _create_random_state_generator(self, base_seed, offset=0):
        """Create a fresh random state generator for isolation."""
        if base_seed is None:
            return None
        # Create a new random state based on base seed + offset
        new_seed = base_seed + offset if base_seed is not None else None
        return np.random.RandomState(new_seed)


    def fit(self, stage_inputs, y):

        self.oob_scores = []
        self.models = []

        if isinstance(stage_inputs, dict):
            input_list = list(stage_inputs.values())
        elif isinstance(stage_inputs, list):
            input_list = stage_inputs
        elif isinstance(stage_inputs, np.ndarray):
            input_list = [stage_inputs]
        else:
            raise ValueError("stage_inputs must be dict, list or numpy array")

        num_inputs = len(input_list)
        
        n_samples, n_features = np.shape(input_list[0])

        # Create a base random state for this training session
        # This ensures each fit() call is independent
        if self.random_state is not None:
            # Use a hashed version of random_state to ensure independence
            base_rng = np.random.RandomState(self.random_state)
            # Generate a unique seed for this training session
            session_seed = base_rng.randint(0, 2**31)
        else:
            session_seed = None
        
        # print("shape of input:", np.shape(oob_decision_function))
        print()
        print(f"number of input list {num_inputs}")
        best_mse = np.inf
        no_improve = 0
        rng = np.random.RandomState(self.random_state)

        X = None
        Layers_output = []

        for layer in range(self.max_layers):
            if layer == 0:
                current_input_idx = 0
                current_input = input_list[current_input_idx]
                X_layer_input = current_input.copy()
            else:
                current_input_idx = (layer - 1) % num_inputs
                current_input = input_list[current_input_idx]
                X_layer_input = np.hstack([X, current_input])

            oob_decision_function = np.zeros((n_samples, 1))


            
            print("----------------------------------")
            print(f"Label: {y[0]}")
            print(f"X_train at this round {X_layer_input[0]}")

            if self.verbose:
                print(f"[Layer {layer}] Input shape: {X_layer_input.shape}, from stage {current_input_idx}")

            layer_models = []
            layer_preds = []
            mse_accum = []

            # global_rng = np.random.RandomState(self.random_state)

            for est in range(self.n_estimators):

                if session_seed is not None:
                    # Use different seed generation that doesn't accumulate state
                    rf_seed = session_seed + 1000 * layer + 10 * est
                    erf_seed = session_seed + 1000 * layer + 10 * est + 1
                else:
                    rf_seed = None
                    erf_seed = None

                rf = RandomForestRegressor(
                    n_estimators=self.n_trees,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    criterion=self.criterion,
                    n_jobs=self.n_jobs,
                    oob_score=(self.validation_mode == 'oob'),
                    bootstrap=True,
                    random_state=rf_seed
                )

                erf = ExtraTreesRegressor(
                    n_estimators=self.n_trees,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    criterion=self.criterion,
                    n_jobs=self.n_jobs,
                    oob_score=(self.validation_mode == 'oob'),
                    bootstrap=True,
                    random_state=erf_seed
                )
                
                # est_seed + 1 if est_seed is not None else None

                x_aug_rf = self.fit_transform(X_layer_input, y, rf)
                x_aug_erf = self.fit_transform(X_layer_input, y, erf)

                oob_decision_function += self.oob_decision_function_(rf)
                oob_decision_function += self.oob_decision_function_(erf)


                layer_models.append((rf, erf))


                layer_preds.append(x_aug_rf)
                layer_preds.append(x_aug_erf)

                if self.verbose:
                    print(f"  Estimator pair {est}:  Output shape: {np.hstack(layer_preds[-2:]).shape}")

            X_new_features = np.hstack(layer_preds)
            X = X_new_features.copy()
            Layers_output.append(X)
            print(f"Layer first predition: {X[0]}")

            self.oob_decision_ = oob_decision_function / (2 * self.n_estimators)
            y_pred = self.oob_decision_
            avg_mse = mean_squared_error(y, y_pred)

            # avg_mse = np.mean(mse_accum)
            self.oob_scores.append(avg_mse)

            if self.verbose:
                print(f"[Layer {layer}] Output features: {X.shape}, Avg. validation MSE: {avg_mse:.6f}")

            if avg_mse + self.delta < best_mse:
                # Improvement found
                best_mse = avg_mse
                no_improve = 0
                # self.models.append(layer_models)
            else:
                # No improvement
                no_improve += 1
                if self.verbose:
                    print(f"[Layer {layer}] No improvement count: {no_improve}/{self.n_tolerant_rounds}")

            # ✅ Always append the current layer before checking stopping condition
            self.models.append(layer_models)

            # Now check for early stopping
            if no_improve >= self.n_tolerant_rounds:
                if self.verbose:
                    print(f"Early stopping at layer {layer}.")
                    # print(f"✅ Optimal number of layers: {len(self.models)}")
                    # print(f"shape of the savect models:{np.shape(self.models)}")
                    print(f"✅ Optimal number of layers: {len(self.models) - self.n_tolerant_rounds}")
                    
                # Roll back the last `n_tolerant_rounds` layers (bad ones) if desired
                self.models = self.models[:-self.n_tolerant_rounds]
                Layers_output = Layers_output[:-self.n_tolerant_rounds]
                print(f"total layers feature shape: {[np.shape(layer) for layer in Layers_output]}")
                break
            
                

        self.final_features_ = Layers_output[-1] if Layers_output else None

        if self.use_predictor:
            if self.verbose:
                print(f"Training final predictor: {self.predictor_name.upper()}")

            if self.predictor_name == "rf":
                self.final_predictor = RandomForestRegressor(
                    n_estimators=self.n_trees,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            elif self.predictor_name == "xgboost":
                try:
                    from xgboost import XGBRegressor
                except ImportError:
                    raise ImportError("XGBoost is not installed. Please run `pip install xgboost`.")
                self.final_predictor = XGBRegressor(
                    n_estimators=self.n_trees,
                    max_depth=self.max_depth or 6,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbosity=0
                )
            elif self.predictor_name == "lightgbm":
                try:
                    from lightgbm import LGBMRegressor
                except ImportError:
                    raise ImportError("LightGBM is not installed. Please run `pip install lightgbm`.")
                self.final_predictor = LGBMRegressor(
                    n_estimators=self.n_trees,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            else:
                raise ValueError("Unsupported predictor name. Use 'rf', 'xgboost', or 'lightgbm'.")

            self.final_predictor.fit(self.final_features_, y)

    def feature_importances_(self):
        if not self.models:
            raise ValueError("No models found. Please fit the model before accessing feature importances.")

        # Aggregate feature importances from all layers and estimators
        all_importances = {}
        for layer_idx, layer_models in enumerate(self.models):
            layer_imps = []
            for rf, erf in layer_models:
                layer_imps.append(rf.feature_importances_)
                layer_imps.append(erf.feature_importances_)

            layer_imps = np.vstack(layer_imps) 
            all_importances[layer_idx] = layer_imps

        return all_importances

    def predict(self, stage_inputs):
        if isinstance(stage_inputs, dict):
            input_list = list(stage_inputs.values())
        elif isinstance(stage_inputs, list):
            input_list = stage_inputs
        elif isinstance(stage_inputs, np.ndarray):
            input_list = [stage_inputs]
        else:
            raise ValueError("stage_inputs must be dict, list or numpy array")

        num_inputs = len(input_list)
        X = None

        print()
        print("...........")
        for layer_idx, layer_models in enumerate(self.models):
            if layer_idx == 0:
                current_input = input_list[0]
                X_layer_input = current_input.copy()
            else:
                current_input_idx = (layer_idx - 1) % num_inputs
                current_input = input_list[current_input_idx]
                X_layer_input = np.hstack([X, current_input])

            if self.verbose:
                print(f"[Predict Layer {layer_idx}] Input shape: {X_layer_input.shape}")

            layer_preds = []
            for rf, erf in layer_models:
                layer_preds.append(rf.predict(X_layer_input).reshape(-1, 1))
                layer_preds.append(erf.predict(X_layer_input).reshape(-1, 1))

            X_new_features = np.hstack(layer_preds)
            X = X_new_features.copy()

            if self.verbose:
                print(f"[Predict Layer {layer_idx}] Output shape: {X.shape}")

        if self.use_predictor:
            print("Predicton of final predictor")
            return self.final_predictor.predict(X)
        else:
            return np.mean(np.hstack(layer_preds), axis=1)
        

    def reset(self):
        """
        Fully reset to cold-start state.
        """
        if hasattr(self, "models"):
            for layer in self.models:
                for rf, erf in layer:
                    del rf, erf
            self.models = []

        if hasattr(self, "final_predictor") and self.final_predictor is not None:
            del self.final_predictor
            self.final_predictor = None

        for attr in ["final_features_", "oob_decision_", "oob_scores"]:
            if hasattr(self, attr):
                delattr(self, attr)

        self.oob_scores = []
        
        gc.collect()
        
        if self.verbose:
            print("DFCascadeRegressor has been fully reset.")

