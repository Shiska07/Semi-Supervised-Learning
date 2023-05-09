import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

D_s = pd.read_csv('D_s_SCD1.csv')
D_u = pd.read_csv('D_u_SCD1.csv')

# randomly pick datapoints to generate the same size dataset as D_s
indices = np.random.randint(low = 0, high = D_u.count()[0], size = (D_s.count()[0],))

D_u = D_u.iloc[indices].copy()
D_u.reset_index(drop = True, inplace = True)

# create dictionary to reference index to fingerprint type
fingerprint_name = {0: 'Topological fingerprints',
                    1: 'Morgan fingerprints',
                    2: 'Atompair fingerprints',
                    3: 'Topological Torsion fingerprints'}

def get_fp_arr(fp):
    array = np.zeros((0,), dtype = bool)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array

def get_topological_fp(mol):
    fpgen = AllChem.GetRDKitFPGenerator(maxPath=3, fpSize=1024)
    return get_fp_arr(fpgen.GetFingerprint(mol))

def get_morgan_fp(mol):
    fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=300)
    return get_fp_arr(fpgen.GetFingerprint(mol))

def get_atompair_fp(mol):
    fpgen = AllChem.GetAtomPairGenerator(countSimulation=False, fpSize=1024)
    return get_fp_arr(fpgen.GetFingerprint(mol))

def get_toptorsion_fp(mol):
    fpgen = AllChem.GetTopologicalTorsionGenerator(countSimulation=False, fpSize=1024)
    return get_fp_arr(fpgen.GetFingerprint(mol))


# takes a series of smiles and returns a list containing
# 4 different dataframes for 4 different fingerprints
def get_fingerprints(smiles_ser):
    fp_list = []

    # convert series to list
    smiles_list = smiles_ser.tolist()

    # convert smiles to mols
    mols_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]

    # get topological fingerprints
    top_list = [get_topological_fp(mol) for mol in mols_list]
    top_df = pd.DataFrame(np.array(top_list, dtype=bool))
    fp_list.append(top_df)

    # get morgan fingerprints
    mor_list = [get_morgan_fp(mol) for mol in mols_list]
    mor_df = pd.DataFrame(np.array(mor_list, dtype=bool))
    fp_list.append(mor_df)

    # atompair fingerprints
    ap_list = [get_atompair_fp(mol) for mol in mols_list]
    ap_df = pd.DataFrame(np.array(ap_list, dtype=bool))
    fp_list.append(ap_df)

    # topological torsion
    tor_list = [get_toptorsion_fp(mol) for mol in mols_list]
    tor_df = pd.DataFrame(np.array(tor_list, dtype=bool))
    fp_list.append(tor_df)

    return fp_list

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(D_s['SMILES'], D_s['activity'],
                                                    test_size=0.25, stratify = D_s['activity'])

X_train_fps_s = get_fingerprints(X_train)
X_test_fps = get_fingerprints(X_test)
X_train_fps_u = get_fingerprints(D_u['SMILES'])
y_train_all_views = [y_train, y_train, y_train, y_train]


def get_best_params():
    # random gird will be for hyperparameter tuning
    random_grid = {
        'n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'weights': ['uniform', 'distance'],
        'metric': ['jaccard', 'dice']
    }

    # cross validation to find good hyperparameters
    cv_params = []
    knn_c = KNeighborsClassifier()
    for i, X_train_fp in enumerate(X_train_fps_s):
        knn_cv = RandomizedSearchCV(estimator=knn_c, param_distributions=random_grid, n_iter=10, verbose=1)
        knn_cv.fit(X_train_fp, y_train)
        print(f'{knn_cv.best_params_}\n')
        cv_params.append(knn_cv.best_params_)

    return cv_params


# trains the model on labeled data
def train_models(cv_params):

    # fit final models
    final_models = []
    for i, params in enumerate(cv_params):
        knn_final = KNeighborsClassifier(**params)
        knn_final.fit(X_train_fps_s[i], y_train)
        final_models.append(knn_final)

    return final_models


# gets indices and prediction for 'n' most confident predictions in each view
def get_best_n_prediction_indices(knn_models, n):
    # initialize list to store most confident indices and labels for each view
    indices_list = []
    labels_list = []

    for i, val in enumerate(knn_models):
        # get number of samples in unlabeled data
        n_samples = X_train_fps_u[i].shape[0]

        # get predictions and prediction probability
        y_pred_u = knn_models[i].predict(X_train_fps_u[i])
        y_pred_prob_u = knn_models[i].predict_proba((X_train_fps_u[i]))[np.arange(0, n_samples), y_pred_u]

        # get indices and labels of datapoints with most confident predictions
        max_prob_indices = np.argsort(y_pred_prob_u)[::-1][:n]
        max_prob_labels = y_pred_u[max_prob_indices]

        indices_list.append(max_prob_indices)
        labels_list.append(max_prob_labels)

    return indices_list, labels_list


# returns min of the total number of unlabeled samples remaining in each view
def get_min_unlabeled_data_count():
    samples_count_list = [df.count()[0] for df in X_train_fps_u]
    return min(samples_count_list)


# given a list of indices and predicted labels for each view from the unlabeled data
# adds datapoints to the training data
# predcitions made by one model in a single view are added to training data in all other views
def add_unlabeled_data(indices_list, labels_list):
    for i in range(len(indices_list)):
        # make copy
        indices_to_use = indices_list.copy()
        labels_to_use = labels_list.copy()

        # remove ith indices list and labels list
        indices_to_use.pop(i)
        labels_to_use.pop(i)

        # create a single array
        indices_arr = (np.array(indices_to_use)).flatten()
        labels_arr = (np.array(labels_to_use)).flatten()

        # create dataframe to concatenate
        smiles_ser = (X_train_fps_u[i].iloc[indices_arr]).reset_index(drop=True)
        labels_ser = pd.Series(labels_arr, name='acitvity')
        df_to_add = pd.concat([smiles_ser, labels_ser], axis=1)

        # drop duplicates and reset index
        df_to_add.drop_duplicates(subset=df_to_add.columns[:-1], inplace=True)
        df_to_add.reset_index(inplace=True, drop=True)

        # concatenate to labeled data and reset indices
        X_train_fps_s[i] = pd.concat([X_train_fps_s[i], df_to_add[df_to_add.columns[:-1]]], axis=0)
        X_train_fps_s[i].reset_index(inplace=True, drop=True)

        y_train_all_views[i] = pd.concat([y_train_all_views[i], df_to_add['acitvity']])
        y_train_all_views[i].reset_index(inplace=True, drop=True)

        # remove added data from unlabeled df
        X_train_fps_u[i].drop(index=indices_arr, axis=0, inplace=True)

    return 0

def get_combined_test_accuracy(models_list):
    n_test_samples = y_test.shape[0]
    y_pred_values = np.zeros((n_test_samples, len(models_list)), dtype=int)
    y_pred_final = np.zeros(n_test_samples)

    # get predictions for each model
    for i, model in enumerate(models_list):
        y_pred_values[:, i] = model.predict(X_test_fps[i])

    # most frequent prediction is the final prediction
    for i in range(n_test_samples):
        # predictions for a single sample by all 4 models
        y_preds_i = y_pred_values[i, :]

        # get counts for all labels
        values, counts = np.unique(y_preds_i, return_counts=True)

        # index of label with the highest counts
        mode_idx = np.argmax(counts)

        y_pred_final[i] = values[mode_idx]

    # get accuracy
    acc = accuracy_score(y_test, y_pred_final)
    f1 = f1_score(y_test, y_pred_final)

    return acc, f1


# prints test accuracy values given predictions and true value
def print_individual_test_accuracies(models_list):
    # get prediction and accuracy for each model
    for i, model in enumerate(models_list):
        y_pred = model.predict(X_test_fps[i])
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f'Accuracy for model trained on {fingerprint_name[i]} is {acc:0.3f} and f1-score is {f1:0.3f}.\n')

# train models for each view on the labeled data
model_params = get_best_params()
sup_models = train_models(model_params)

idx_list, lab_list = get_best_n_prediction_indices(sup_models, 10)

res = print_combined_test_accuracy(sup_models)
