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


def get_fp_arr(fp):
    array = np.zeros((0,), dtype = int)
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
    top_df = pd.DataFrame(np.array(top_list, dtype=int))
    fp_list.append(top_df)

    # get morgan fingerprints
    mor_list = [get_morgan_fp(mol) for mol in mols_list]
    mor_df = pd.DataFrame(np.array(mor_list, dtype=int))
    fp_list.append(mor_df)

    # atompair fingerprints
    ap_list = [get_atompair_fp(mol) for mol in mols_list]
    ap_df = pd.DataFrame(np.array(ap_list, dtype=int))
    fp_list.append(ap_df)

    # topological torsion
    tor_list = [get_toptorsion_fp(mol) for mol in mols_list]
    tor_df = pd.DataFrame(np.array(tor_list, dtype=int))
    fp_list.append(tor_df)

    return fp_list

fps = get_fingerprints(D_s['SMILES'])
