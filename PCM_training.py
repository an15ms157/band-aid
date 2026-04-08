import time
import os 
import sys
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, make_scorer, confusion_matrix
import optuna
import optuna.visualization.matplotlib as vis_mat
from optuna.pruners import MedianPruner
from hipe4ml import plot_utils
import shap
import matplotlib

# Set up pandas and matplotlib globally
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
matplotlib.use('Agg')

# Ensure helper_functions can be imported by adding src directory to path
src_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from helper_functions import bdt_utils, weight_function


def run_xgb_training(pTmin=13.0, pTmax=14.0, n_DEBUG=1000, b_OPTUNA=False, Keyword="_OldFeatures_NewBackground", max_EPOCH=3):

    pT_bin_STR = f"{pTmin} <Pt< {pTmax}"
    pT_bin_str = f"{pTmin}_{pTmax}"

    base_dir = f"/misc/alidata150/alice_u/nath/ML4PCM/model_training/src/XGB_MBNormWeightedMC_noPTfeatures" 
    output_dir = f"{base_dir}/{Keyword}/temp/{pTmin}-{pTmax}" 
    central_temp_dir = "/misc/alidata150/alice_u/nath/ML4PCM/Parquet_DEC2025/training/src/Central_XGB_Photon_Training/temp"
    plots_dir = f"{central_temp_dir}/{pT_bin_str}"
    print(f"Output directory: {output_dir}")
    print(f"Plots directory: {plots_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    os.chdir(output_dir)
    log_file = f"{output_dir}/run_{pTmin}-{pTmax}.log"


    # Clear previous handlers if any (to avoid duplicates)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Overwrite log file each run
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)  

    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.enable_default_handler()
    optuna.logging.enable_propagation()


    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(pT_bin_STR)
    logger.info(f"logging at : {os.path.abspath(log_file)}")


    # File locations
    MC_Parquet_location_AS2 =   f"/misc/alidata150/alice_u/nath/ML4PCM/Parquet_DEC2025/Combined/combined_24a1a2/{pT_bin_str}/Photon_GCo_3000_MC_TreeForPhotonML_10130023_0d200008100000001100400000_0152101500000000.parquet"
    MC_Parquet_location_AS1 =   f"/misc/alidata150/alice_u/nath/ML4PCM/Parquet_DEC2025/Combined/combined_20g10/{pT_bin_str}/Photon_GCo_3000_MC_TreeForPhotonML_10130023_0d200008100000001100400000_0152101500000000.parquet"
    MC_Parquet_location_MB  =   f"/misc/alidata150/alice_u/nath/ML4PCM/Parquet_DEC2025/Combined/combined_20e3/{pT_bin_str}/Photon_GCo_3000_MC_TreeForPhotonML_10130023_0d200008100000001100400000_0152101500000000.parquet"

    # Parallel parquet loading (30-50% faster than sequential)
    logger.info("Loading parquet files in parallel...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_AS2 = executor.submit(pd.read_parquet, MC_Parquet_location_AS2)
        future_AS1 = executor.submit(pd.read_parquet, MC_Parquet_location_AS1)
        future_MB = executor.submit(pd.read_parquet, MC_Parquet_location_MB)
        MC_AS2 = future_AS2.result()
        MC_AS1 = future_AS1.result()
        MC_MB = future_MB.result()
    
    # Check if columns are the same across all dataframes
    same_columns = (list(MC_AS2.columns) == list(MC_AS1.columns) == list(MC_MB.columns))
    logger.info(f"All dataframes have the same columns: {same_columns}")

    # pT Filter
    MC_AS2 = MC_AS2.query(pT_bin_STR)
    MC_AS1 = MC_AS1.query(pT_bin_STR)
    MC_MB = MC_MB.query(pT_bin_STR)



    ##############################################
###############         Assign MC weights    ##############
    #################################################
    
# Weights assignment block using weight functions

    if pTmax <= 0.2:
        MC_MB['mc_weight'] = 1.0
        MC_AS1['mc_weight'] = 1.0
        MC_AS2['mc_weight'] = 1.0
        logger.info("Using constant weights for low pT (< 0.2 GeV)")
    elif pTmax <= 7.0:
        MC_MB['mc_weight'] = 1.0
        # Apply weight functions based on Pt values
        funcWeightAS1 = weight_function.weight_function_MB_AS1(MC_AS1['Pt'].values)
        funcWeightAS2 = weight_function.weight_function_MB_AS2(MC_AS2['Pt'].values)
        MC_AS1['mc_weight'] = funcWeightAS1/weight_function.weight_function_MB_AS1(pTmin)
        MC_AS2['mc_weight'] = funcWeightAS2/weight_function.weight_function_MB_AS2(pTmin)

        logger.info("Using weight functions for intermediate pT ( [0.4-7.0] GeV)")
        logger.info(f"AS1 weights range: {funcWeightAS1.min():.4f} - {funcWeightAS1.max():.4f}")
        logger.info(f"AS2 weights range: {funcWeightAS2.min():.4f} - {funcWeightAS2.max():.4f}")
    else:
        MC_MB['mc_weight'] = 1.0
        funcWeightAS1 = weight_function.weight_function_MB_AS1(MC_AS1['Pt'].values)
        MC_AS1['mc_weight'] = funcWeightAS1/weight_function.weight_function_MB_AS1(pTmin)
        MC_AS2['mc_weight'] = 0.0
        logger.info("Using weight function for AS1 only for high pT (< 7.0 GeV), AS1 weight = 0")
        logger.info(f"AS1 weights range: {funcWeightAS1.min():.4f} - {funcWeightAS1.max():.4f}")

    # Get sample counts
    n_MB = len(MC_MB)
    n_AS1 = len(MC_AS1) 
    n_AS2 = len(MC_AS2)

    logger.info(
        "Weights for MC samples:\n"
        f"#MB, #AS1, #AS2: {n_MB}, {n_AS1}, {n_AS2}\n"
        f"MB weight: constant 1.0\n"
        "Generating quick weight verification plot..."
    )

    # Quick plot of pT*mc_weight distribution for epoch=1 - Publication Ready
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(MC_MB['Pt'], bins=100, weights=MC_MB['mc_weight'], alpha=0.6, label='MB', density=True, histtype='step', log=True, linewidth=2.5)
    ax.hist(MC_AS1['Pt'], bins=100, weights=MC_AS1['mc_weight'], alpha=0.6, label='AS1', density=True, histtype='step', log=True, linewidth=2.5)
    ax.hist(MC_AS2['Pt'], bins=100, weights=MC_AS2['mc_weight'], alpha=0.6, label='AS2', density=True, histtype='step', log=True, linewidth=2.5)
    ax.set_xlabel('$p_\\mathrm{T}$ (GeV/$c$)', fontsize=14, fontweight='bold')
    ax.set_xlim(pTmin, pTmax)
    ax.set_ylabel('Density (a.u.)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    ax.set_title('$p_\\mathrm{T}$ × MC Weight Distribution by Sample', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pt_weight_distribution.png'), bbox_inches="tight", dpi=300)
    plt.close()
    logger.info("pt_weight_distribution plot complete")

    ##############################################
###############    Filtering sig/bkg    ##############
    #################################################

    # Filter -> pT & kind
    conditions_Signal_Photon = "kind < 1 and " + pT_bin_STR
    conditions_Background_Photon = "kind >= 1 and " + pT_bin_STR

    Signal_Photon_MB = MC_MB.query(conditions_Signal_Photon)
    Signal_Photon_AS1 = MC_AS1.query(conditions_Signal_Photon)
    Signal_Photon_AS2 = MC_AS2.query(conditions_Signal_Photon)

    Background_Photon_MB = MC_MB.query(conditions_Background_Photon)
    Background_Photon_AS1 = MC_AS1.query(conditions_Background_Photon)
    Background_Photon_AS2 = MC_AS2.query(conditions_Background_Photon)

    logger.info(
        f"Signal_Photon_MB: {len(Signal_Photon_MB)} "
        f"Signal_Photon_AS1: {len(Signal_Photon_AS1)} "
        f"Signal_Photon_AS2: {len(Signal_Photon_AS2)} "
        f"Background_Photon_MB: {len(Background_Photon_MB)} "
        f"Background_Photon_AS1: {len(Background_Photon_AS1)} "
        f"Background_Photon_AS2: {len(Background_Photon_AS2)}"
    )



    # Concatenate signal and background with labels 
    if n_DEBUG:
        logger.info(f"DEBUG mode is ON, using a subset {n_DEBUG} of the data for training.\n")
        Photon_Signal_Pt = Signal_Photon_MB
        Photon_Background_Pt = Background_Photon_MB
        logger.info("Using MB signal and MB background")
        logger.info(f"#Signal: {Photon_Signal_Pt.tail(10)} \n{Photon_Signal_Pt.head(10)}")
        logger.info(f"#Background: {Photon_Background_Pt.tail(10)} \n{Photon_Background_Pt.head(10)}")
    elif pTmin >= 7.0:
        Photon_Signal_Pt = pd.concat([Signal_Photon_MB, Signal_Photon_AS1], ignore_index=True)
        Photon_Background_Pt = pd.concat([Background_Photon_MB, Background_Photon_AS1], ignore_index=True)
        logger.info("Using MB+AS1 signal and MB+AS1 background")
        logger.info(f"#Signal: {Photon_Signal_Pt.tail(10)} \n{Photon_Signal_Pt.head(10)}")
        logger.info(f"#Background: {Photon_Background_Pt.tail(10)} \n{Photon_Background_Pt.head(10)}")
    else:
        Photon_Signal_Pt = pd.concat([Signal_Photon_MB, Signal_Photon_AS1, Signal_Photon_AS2], ignore_index=True)
        Photon_Background_Pt = pd.concat([Background_Photon_MB, Background_Photon_AS1, Background_Photon_AS2], ignore_index=True)
        logger.info("Using MB+AS1+AS2 signal and MB+AS1+AS2 background")
        logger.info(f"#Signal: {Photon_Signal_Pt.tail(10)} \n{Photon_Signal_Pt.head(10)}")
        logger.info(f"#Background: {Photon_Background_Pt.tail(10)} \n{Photon_Background_Pt.head(10)}")


    Photon_Signal_Pt = Photon_Signal_Pt.copy()
    Photon_Background_Pt = Photon_Background_Pt.copy()

    Wsig = Photon_Signal_Pt["mc_weight"].sum()
    Wbkg = Photon_Background_Pt["mc_weight"].sum()

    scale_pos_weight = Wbkg / Wsig


    logger.info(f"Weighted Signal (used in X): {Photon_Signal_Pt['mc_weight'].sum()}")
    logger.info(f"Weighted Background (used in X): {Photon_Background_Pt['mc_weight'].sum()}")
    logger.info(f"#Signal: {Photon_Signal_Pt.tail(10)} \n{Photon_Signal_Pt.head(10)}")
    logger.info(f"#Background: {Photon_Background_Pt.tail(10)} \n{Photon_Background_Pt.head(10)}")



    X = pd.concat([Photon_Signal_Pt, Photon_Background_Pt], ignore_index=True)
    y = np.concatenate([
        np.ones(len(Photon_Signal_Pt)),
        np.zeros(len(Photon_Background_Pt))
    ])
    y = pd.Series(y, index=X.index)

    # Extract and remove mc_weight column
    sample_weights = X['mc_weight'].values
    sample_weights = pd.Series(sample_weights, index=X.index)
    
    # Log weighted sums before dropping mc_weight
    Wsig_extracted = float(X.loc[y==1, 'mc_weight'].sum())
    Wbkg_extracted = float(X.loc[y==0, 'mc_weight'].sum())
    logger.info(f"SumW(sig), SumW(bkg) in X: {Wsig_extracted:.2f}, {Wbkg_extracted:.2f}")
    
    X = X.drop(columns=['mc_weight'])


    
        ##############################################
###############          EPOCH sig/bkg       ##############
        #############################################

    if max_EPOCH >= 8:
        N_optuna_sample = min(100000, int(0.5 * len(y)))
        N_optuna    = 100
        T_optuna    = 3600
    elif max_EPOCH >= 6:
        N_optuna_sample = min(10000, int(0.1 * len(y)))
        N_optuna    = 30
        T_optuna    = 100
    else:
        N_optuna_sample = min(10000, int(0.01 * len(y)))
        N_optuna    = 50
        T_optuna    = 200

    #scale_pos_weight = len(Photon_Background_Pt) / len(Photon_Signal_Pt)
    #scale_pos_weight = len(Photon_Signal_Pt) / len(Photon_Background_Pt) 

    logger.info(f"scale_pos_weight (Background/Signal ratio): {scale_pos_weight:.4f}")

    logger.info(f"Sample taken for Optuna (HPO) ~ {N_optuna_sample}, Full training sample ~ {len(y)}")

    if n_DEBUG:
        X = X.sample(n=n_DEBUG, random_state=42)
        y = pd.Series(y, index=X.index)
        sample_weights = pd.Series(sample_weights, index=X.index)

        

    vars_to_draw=MC_MB.columns
    leg_labels = ['signal','background']
    # Correlation plots
    plt.figure(figsize=(10, 10))
    plot_utils.plot_corr([Photon_Signal_Pt], vars_to_draw, [leg_labels[0]])
    plt.savefig(os.path.join(plots_dir, 'photons_source_correlation'), bbox_inches="tight")
    plt.close()
    logger.info("photons_source_correlation plot complete")
    plt.figure(figsize=(10, 10))
    plot_utils.plot_corr([Photon_Signal_Pt], vars_to_draw, [leg_labels[1]])
    plt.savefig(os.path.join(plots_dir, 'photons_background_correlation'), bbox_inches = "tight")
    plt.close()
    logger.info("photons_background_correlation plot complete")

    # Signal and Background Feature Distribution Plots (only for max_EPOCH > 7)
    if max_EPOCH >= 5:
        logger.info("Generating signal and background feature distribution plots...")
        plot_utils.plot_distr(
            [Photon_Background_Pt, Photon_Signal_Pt], 
            vars_to_draw, 
            bins=100, 
            labels=leg_labels, 
            log=True, 
            density=True, 
            figsize=(12, 7), 
            alpha=0.3, 
            grid=False
        )
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
        plt.savefig(os.path.join(plots_dir, 'signal_background_distributions.png'), bbox_inches="tight", dpi=300)
        plt.close()
        logger.info("signal_background_distributions plot complete")

    # Columns to set to -1 (keep them in training)
    #base_exclude =['EPos_Pt', 'ENeg_Pt', 'EPos_Asymmetry']
    base_exclude = [ 'ENeg_ITS_dEdX', 'ENeg_TPC_dEdX', 'ENeg_TOF', 'piMinus_ITS', 'piMinus_TPC', 'piMinus_TOF', 'ENeg_Pt', 'ENeg_Eta', 'EPos_ITS_dEdX', 'EPos_TPC_dEdX', 'EPos_TOF', 'piPlus_ITS', 'piPlus_TPC', 'piPlus_TOF', 'EPos_Pt', 'EPos_Eta', 'EPos_Asymmetry']
    # All daughter Pt features are kept but set to -1 [TECHNICALLY EXCLUDED]
    for col in base_exclude:
        if col in X.columns:
            X[col] = -1
    # Select features (keep all columns, including those set to -1)
    features_for_train_withKind = X.columns
    features_for_train = features_for_train_withKind.drop(['InvMass', 'Pt', 'kind'])  # Production feature set 
    logger.info(f"Features for training: {features_for_train}")

    X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
        X, 
        y, 
        sample_weights, 
        test_size=0.4, #.15 
        random_state=42, 
        stratify=y
    )

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_temp, 
        y_temp, 
        w_temp, 
        test_size=0.25, 
        random_state=42, 
        stratify=y_temp
    )

    logger.info(f"#[Train_Xy, Test_Xy]: {len(X_train)}, {len(y_train)}, {len(X_val)},  {len(y_val)}")
    logger.info(f"X,y type    :   {type(X_train)}, {type(y_train)}")

    # Apply feature selection
    X_train_sel = X_train[features_for_train]
    X_val_sel = X_val[features_for_train]
    X_test_sel = X_test[features_for_train]
    logger.info(f"Selected training dataset': \n {X_train_sel.head(20)}")    
    logger.info(f"Selected validation dataset': \n {X_val_sel.head(20)}")    
    logger.info(f"Selected test dataset': \n {X_test_sel.head(20)}")    

    X_train_sel_withKind = X_train[features_for_train_withKind]
    X_val_sel_withKind = X_val[features_for_train_withKind]
    X_test_sel_withKind = X_test[features_for_train_withKind]
    logger.info("dataset ready after feature selection")

    def objective(trial):
        params = {
            "n_estimators": 800,
            "max_depth": trial.suggest_int("max_depth", 2, 6),  # keep
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.7),
            "gamma": trial.suggest_float("gamma", 0.0, 0.8),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
        }

        # K-Fold Cross-Validation with Stratified sampling to preserve class ratio
        # Ensure train_size is less than dataset size to leave room for test set
        n_samples = len(X_train_sel)
        max_train_size = max(int(0.9 * n_samples), 1)  # Use at most 90% for train, minimum 1
        actual_train_size = min(N_optuna_sample, max_train_size)
        
        X_sampled, _, y_sampled, _, w_sampled, _ = train_test_split(
            X_train_sel, y_train, w_train,
            train_size=actual_train_size,
            stratify=y_train,
            random_state=trial.number
        )

        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=trial.number)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_sampled, y_sampled)):
            X_train_fold = X_sampled.iloc[train_idx]
            X_val_fold = X_sampled.iloc[val_idx]
            y_train_fold = y_sampled.iloc[train_idx]
            y_val_fold = y_sampled.iloc[val_idx]
            w_train_fold = w_sampled.iloc[train_idx]
            w_val_fold = w_sampled.iloc[val_idx]

            clf_fold = xgb.XGBClassifier(**params)
            clf_fold.fit(
                X_train_fold, y_train_fold,
                sample_weight=w_train_fold,
                eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
                sample_weight_eval_set=[w_train_fold, w_val_fold],
                eval_metric="auc",
                callbacks=[xgb.callback.EarlyStopping(
                    rounds=20,  # Aggressive stopping with smaller n_estimators
                    data_name="validation_1",
                    metric_name="auc",
                    save_best=True
                )],
                verbose=False
            )

            y_pred_proba = clf_fold.predict_proba(
                X_val_fold,
                iteration_range=(0, clf_fold.best_iteration + 1)
            )[:, 1]
            auc = roc_auc_score(y_val_fold, y_pred_proba, sample_weight=w_val_fold)
            scores.append(auc)
            mean_auc = np.mean(scores)
            
            # Report intermediate value for Optuna pruning
            trial.report(float(mean_auc), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
        std_auc  = np.std(scores)
        mean_auc = np.mean(scores)
        trial.set_user_attr("auc_mean", mean_auc)
        trial.set_user_attr("auc_std", std_auc)

        return mean_auc




    if b_OPTUNA:
        optuna_db_path = f"sqlite:///{os.path.join(plots_dir, f'xgb_opt_{pTmin}-{pTmax}.db')}"
        study = optuna.create_study(
            direction="maximize",  
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0), 
            study_name="xgb_opt", 
            storage=optuna_db_path, 
            load_if_exists=True
            )
        study.optimize(objective, n_trials=N_optuna, show_progress_bar=True, timeout=T_optuna, n_jobs=3)
        logger.info(f"Best hyperparameters: {study.best_params}")

        logger.info(f"Optuna study direction: {study.direction}")
        logger.info(f"Number of completed trials: {len(study.trials)}")
        logger.info(f"Best trial value: {study.best_trial.value}")
        logger.info(f"Best trial params: {study.best_trial.params}")
        logger.info("All trial results:")

        def _fmt_float(value, digits: int = 5):
            if value is None:
                return "None"
            return f"{float(value):.{digits}f}"

        for t in study.trials:
            logger.info(
                f"Trial {t.number}: "
                f"value={_fmt_float(t.value)}, "
                f"auc_mean={_fmt_float(t.user_attrs.get('auc_mean'))}, "
                f"auc_std={_fmt_float(t.user_attrs.get('auc_std'))}, "
                f"params={t.params}, "
                f"state={t.state}"
            )

        # Optuna visualization plots using matplotlib
        fig1 = vis_mat.plot_param_importances(study)
        plt.savefig(os.path.join(plots_dir, "optuna_param_importance.png"), bbox_inches="tight")
        plt.close()
        logger.info("optuna_param_importance plot complete")
        fig2 = vis_mat.plot_optimization_history(study)
        plt.savefig(os.path.join(plots_dir, "optuna_optimization_history.png"), bbox_inches="tight")
        plt.close()
        logger.info("optuna_optimization_history plot complete")
        fig3 = vis_mat.plot_slice(study)
        plt.savefig(os.path.join(plots_dir, "optuna_slice_plot.png"), bbox_inches="tight")
        plt.close()
        logger.info("optuna_slice_plot complete")
        fig4 = vis_mat.plot_edf(study)
        plt.savefig(os.path.join(plots_dir, "optuna_edf_plot.png"), bbox_inches="tight")
        plt.close()
        logger.info("optuna_edf_plot complete")
        fig5 = vis_mat.plot_terminator_improvement(study)
        plt.savefig(os.path.join(plots_dir, "optuna_terminator_improvement.png"), bbox_inches="tight")
        plt.close()
        logger.info("optuna_terminator_improvement plot complete")
        plt.figure(figsize=(15, 15))  # Add this line before plotting
        fig6 = vis_mat.plot_contour(study)
        plt.savefig(os.path.join(plots_dir, "optuna_contour_plot.png"), bbox_inches="tight")
        plt.close()
        logger.info("optuna_contour_plot complete")

        best_params = study.best_params
        best_params.update({
            "n_estimators": 1000,
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist"  
        })
        final_model = xgb.XGBClassifier(**best_params)
    else:
        final_model = xgb.XGBClassifier(
            n_estimators=1000,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist"
        )

    logger.info(f"LAST CHECK :\n {X_train_sel.head(200)}, {y_train[:200]}")
    final_model.fit(
        X_train_sel, y_train,
        sample_weight=w_train,
        eval_set=[(X_train_sel, y_train), (X_val_sel, y_val)],
        sample_weight_eval_set=[w_train, w_val],
        eval_metric="auc",
        callbacks=[xgb.callback.EarlyStopping(
            rounds=30,
            data_name="validation_1",
            metric_name="auc",
            save_best=True
        )],
        verbose=True
    )
    

    logger.info("XGB model trained")

    # Get evaluation results (ALWAYS works)
    evals_result = final_model.evals_result()
    train_auc = np.array(evals_result["validation_0"]["auc"])
    val_auc   = np.array(evals_result["validation_1"]["auc"])

    # Log early stopping information
    if final_model.best_iteration is not None:
        best_iter = final_model.best_iteration
        best_val_auc = val_auc[best_iter]
        best_train_auc = train_auc[best_iter]
        
        logger.info(f"Early stopping triggered at iteration: {best_iter}")
        logger.info(f"Best validation AUC score: {best_val_auc:.6f}")
        logger.info(f"Best train AUC score: {best_train_auc:.6f}")
        logger.info(f"Train-validation gap: {(best_train_auc - best_val_auc):.6f}")
    else:
        logger.info("Early stopping did not trigger - training completed all iterations")
        logger.info(f"Final validation AUC score: {val_auc[-1]:.6f}")
        logger.info(f"Final train AUC score: {train_auc[-1]:.6f}")

    
    # Save model as .model file (XGBoost native format)
    model_filename = "PhotonML_"+str(pTmin)+"_Pt_"+str(pTmax)+"_XGB.model"
    final_model.save_model(os.path.join(plots_dir, model_filename))
    logger.info("XGB model saved")

    # Save model as pickle file as well
    pkl_filename = "PhotonML_"+str(pTmin)+"_Pt_"+str(pTmax)+"_XGB.pkl"
    with open(os.path.join(plots_dir, pkl_filename), 'wb') as f:
        pickle.dump(final_model, f)
    logger.info(f"XGB model saved as pickle: {pkl_filename}")

    # Predicted Labels
    y_pred_train = final_model.predict(X_train_sel)
    y_pred_val = final_model.predict(X_val_sel)
    y_pred_test = final_model.predict(X_test_sel)
    logger.info(f"TEST prediction labels (first N): {y_pred_test[:20]}")

    # Predicted Probabilities (for ROC/PR curves) using best iteration
    y_prob_train = final_model.predict_proba(
        X_train_sel,
        iteration_range=(0, final_model.best_iteration + 1)
    )[:, 1]
    y_prob_val = final_model.predict_proba(
        X_val_sel,
        iteration_range=(0, final_model.best_iteration + 1)
    )[:, 1]
    y_prob_test = final_model.predict_proba(
        X_test_sel,
        iteration_range=(0, final_model.best_iteration + 1)
    )[:, 1]

    logger.info(f"TEST Probabilities (first N): {y_prob_test[:20]}")
    logger.info(f"Checking same entries for labels and probability {len(y_pred_test)}, {len(y_prob_test)}")

    fpr, tpr, thresholds = roc_curve(y_train, y_prob_train, sample_weight=w_train)
    auc_train = roc_auc_score(y_train, y_prob_train, sample_weight=w_train)


    logger.info(f"Train ROC AUC: {auc_train}")
    logger.info(f"Validation ROC AUC: {roc_auc_score(y_val, y_prob_val, sample_weight=w_val)}")
    logger.info(f"Test ROC AUC: {roc_auc_score(y_test, y_prob_test, sample_weight=w_test)}")
    logger.info(f"Classification report (test):\n{classification_report(y_test, y_pred_test)}")

    # Print how many entries are in each probability bin (0-0.1, 0.1-0.2, ..., 0.9-1.0)
    bins = np.arange(0, 1.1, 0.1)
    hist, bin_edges = np.histogram(y_prob_test, bins=bins)
    for i in range(len(hist)):
        logger.info(f"Entries in {bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}: {hist[i]}")

    # After training, evaluate at different thresholds
    logger.info("\n=== Threshold Sweep Analysis ===")
    thresholds = np.linspace(0, 1, 101)
    threshold_results = []
    for th in thresholds:
        y_pred_at_th = (y_prob_test > th).astype(int)
        cm = confusion_matrix(y_test, y_pred_at_th, sample_weight=w_test)
        
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge case where only one class is predicted
            tn, fp, fn, tp = 0, 0, 0, 0
            if cm.size == 1:
                if y_test.iloc[0] == 0:
                    tn = cm[0, 0]
                else:
                    tp = cm[0, 0]
        
        sig_eff = tp / (tp + fn) if (tp + fn) > 0 else 0  # Signal efficiency (recall)
        bkg_rej = tn / (tn + fp) if (tn + fp) > 0 else 0  # Background rejection (specificity)
        purity = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        threshold_results.append({
            'threshold': th,
            'sig_eff': sig_eff,
            'bkg_rej': bkg_rej,
            'purity': purity
        })
        
        logger.info(f"Threshold {th:.2f}: Sig_Eff={sig_eff:.3f}, Bkg_Rej={bkg_rej:.3f}, Purity={purity:.3f}")
    
    # Find optimal threshold (max product of sig_eff and bkg_rej)
    best_idx = np.argmax([r['sig_eff'] * r['bkg_rej'] for r in threshold_results])
    best_result = threshold_results[best_idx]
    logger.info(f"\nOptimal threshold (max Sig_Eff × Bkg_Rej): {best_result['threshold']:.2f}")
    logger.info(f"  Sig_Eff={best_result['sig_eff']:.3f}, Bkg_Rej={best_result['bkg_rej']:.3f}, Purity={best_result['purity']:.3f}")

    if max_EPOCH >= 5:
        # Check feature importances
        #importances = final_model.feature_importances_
        #for name, score in sorted(zip(features_for_train, importances), key=lambda x: -x[1]):
        #    logger.info(f"{name}: {score}")



        bdt_utils.plot_bdt_output(final_model, X_train_sel, y_train, X_test_sel, y_test, w_train, w_test, output_margin=False, output_path=os.path.join(plots_dir, 'bdt_output_train_test.png'))
        logger.info("bdt_output_train_test plot complete")

        bdt_utils.plot_roc_curves(
            y_train, y_prob_train,
            y_val, y_prob_val,
            y_test, y_prob_test,
            w_train=w_train,
            output_path=os.path.join(plots_dir, "roc_train_test.png")
        )
        logger.info("roc_train_test plot complete")

        bdt_utils.plot_pr_curves(
            y_train, y_prob_train,
            y_val, y_prob_val,
            y_test, y_prob_test,
            w_train=w_train,
            w_val=w_val,
            w_test=w_test,
            output_path=os.path.join(plots_dir, "pr_train_val_test.png")
        )
        logger.info("pr_train_val_test plot complete")

        plt.figure(figsize=(10, 10))  
        plot_utils.plot_output_train_test(
            final_model, 
            [X_train_sel, y_train, X_test_sel, y_test], 
            100, 
            True, 
            leg_labels, 
            True, 
            density=False
        )
        plt.savefig(os.path.join(plots_dir, 'TrainingOutput_Photons'), bbox_inches="tight")
        plt.close()
        logger.info("TrainingOutput_Photons plot complete")

    
        # SHAP values only after 3 epochs
    if max_EPOCH >= 5:
        n_sample_size_train = min(100000, max( 5000, int(len(X_train_sel)) ))
        #n_sample_size_test  = min(100000, max( 5000, int(len(X_test_sel)) ))
        n_sample_size_test  = int(len(X_test_sel))
    
        # === SHAP on Test Set ===
        logger.info(f"Plotting test SHAP features with sample size: {n_sample_size_test}")
        X_sample_test = X_test_sel.sample(n=n_sample_size_test, random_state=42)  

        explainer_test = shap.TreeExplainer(final_model, X_sample_test, approximate=True)
        shap_values_test = explainer_test(X_sample_test, check_additivity=False)
    
        plt.figure(figsize=(10, 10))  
        shap.summary_plot(
            shap_values_test,
            X_sample_test,
            plot_type="bar",
            show=False,
            use_log_scale=False
        )
        plt.savefig(os.path.join(plots_dir, 'shap_summary_bar_test.png'), bbox_inches="tight")
        plt.close()
        logger.info("shap_summary_bar TEST plot complete")
    
        # === SHAP on Training Set ===
        logger.info(f"Plotting training SHAP features with sample size: {n_sample_size_train}")
        X_sample_train = X_train_sel.sample(n=n_sample_size_train, random_state=42)  
        
        explainer_train = shap.TreeExplainer(final_model, X_sample_train, approximate=True)
        shap_values_train = explainer_train(X_sample_train, check_additivity=False)
    
        plt.figure(figsize=(10, 10))  
        shap.summary_plot(
            shap_values_train,
            X_sample_train,
            plot_type="bar",
            show=False,
            use_log_scale=False
        )
        plt.savefig(os.path.join(plots_dir, 'shap_summary_bar_training.png'), bbox_inches="tight")
        plt.close()
        logger.info("shap_summary_bar TRAINING plot complete")

        # Define two cohorts: kind == 0 and kind != 0  
        # NOTE: cohorts must match size of X_sample_train, not full y_train!
        cohorts = ["Signal" if val == 1 else "Background" for val in y_train[X_sample_train.index]]
        shap_cohorts = shap_values_train.cohorts(cohorts)

        plt.figure(figsize=(10, 10))  
        shap.plots.bar(
            shap_cohorts.abs.mean(0), 
            max_display=16, 
            show=False
            )
        plt.savefig(os.path.join(plots_dir, 'shap_double_bar_kind.png'), bbox_inches="tight")
        plt.close()
        logger.info("shap_double_bar_kind plot complete")

        plt.figure(figsize=(10, 10))
        shap.plots.scatter(
            shap_values_train[:, "Alpha"], 
            color=shap_values_train[:, "Qt"],
            show=False
            )
        plt.savefig(os.path.join(plots_dir, 'shap_Alpha.png'), bbox_inches="tight")
        plt.close()
        logger.info("shap_Alpha plot complete")

    # Print all hyperparameters of the final model
    logger.info("Final XGBoost model hyperparameters:")
    logger.info(final_model.get_params())


    if max_EPOCH >= 6:
        logger.info("Plotting significance and efficiency metrics as EPOCH >= 6")
        bdt_utils.plot_photon_significance_metrics(
            final_model, 
            X_test_sel, 
            y_test, 
            w_test=w_test, 
            bins=50,
            output_path=os.path.join(plots_dir, 'photon_significance.png')
        )
        logger.info("photon_significance plot complete")
        
        bdt_utils.plot_pcm_efficiencies(
            final_model, 
            X_train_sel,
            y_train,
            X_test_sel, 
            y_test,
            w_train=w_train,
            w_test=w_test, 
            bins=100,
            output_path=os.path.join(plots_dir, 'purity_curve.png')
        )
        logger.info("PCM purity curve plot complete (train vs test)")

    if max_EPOCH >= 10:
        logger.info("Learning curve plot is onway as EPOCH = 5")
        bdt_utils.plot_xgb_learning_curves(
            xgb.XGBClassifier,
            X_train_sel,
            y_train,
            X_test_sel,
            y_test,
            N_sample=10000,
            output_path=os.path.join(plots_dir, 'Learning_curve_AUC_F1.png')
        )

    logger.info("Training and evaluation completed")
    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    os.chdir(base_dir)

if __name__ == "__main__":
    run_xgb_training(
        pTmin=13.0,
        pTmax=14.0,
        n_DEBUG=1000,
        b_OPTUNA=False,
        Keyword="_OldFeatures_NewBackground"
    )
