from rdkit import DataStructs
from rdkit import Chem
import pandas as pd
from standalone_model_numpy import SCScorer
import os
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import re
import math

# 影响面积
def impact_area(subtree):
    if 'children' not in subtree:
        return 1
    else:
        sub_impact = 0
        for i in subtree['children']:
            sub_impact = sub_impact + impact_area(i)
        return sub_impact + 1

# 输入smiles
def similarity(mole_A, mole_B):
    fps1 = Chem.RDKFingerprint(mole_A)
    fps2 = Chem.RDKFingerprint(mole_B)
    return DataStructs.FingerprintSimilarity(fps1, fps2)

# https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_FULL.csv
# 传进smiles
def reactant_confidence(reactant):
    similar_molecule = []
    article_count = []
    count = 0
    data = pd.read_csv('/Users/chuhanshi/Downloads/USPTO_FULL.csv')
    reactions = data['reactions']
    for i in reactions:
        first = i.find('>')+1
        second = i[first:].find('>')+1
        product = i[first:][second:]
        if similarity(reactant,product)>=0.8:
            count = count + 1
            if product in similar_molecule:
                article_count[similar_molecule.index(product)] = article_count[similar_molecule.index(product)]+1
            else:
                similar_molecule.append(product)
                article_count.append(1)
    confidence = 0
    for i in len(similar_molecule):
        confidence = confidence + similarity(reactant, similar_molecule[i])*article_count[i]

    return confidence/count

def atom_number(molecule):
    mol = Chem.MolFromSmiles(molecule)
    formula = CalcMolFormula(mol)
    pattern = re.compile(r'\d+')  # 查找数字
    result = pattern.findall(formula)
    num = 0
    for i in result:
        num = num + int(i)

    return num

# 反应的confidence (传入所有reactant的list和产物)
def confidence(reactant, product):
    all_atom_num = 0
    # 这里能不能只考虑骨架？
    for i in reactant:
        all_atom_num = all_atom_num + atom_number(i)

    confidence = 0
    for i in reactant:
        confidence = confidence + atom_number(i)/all_atom_num*reactant_confidence(i)

    return confidence

# 反应的confidence (传入所有的reactant的list和产物)
def complexity_reduction(reactant, product):
    project_root = ""
    model = SCScorer()
    model.restore(
        os.path.join(project_root, 'models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.json.gz'))
    score_reactant = 1
    for i in reactant:
        SC = model.get_score_from_smi(i)
        simi = 1 - (SC-1)/4
        score_reactant = score_reactant * simi

    SC_product = model.get_score_from_smi(product)
    score_product = 1 - (SC_product-1)/4

    return score_reactant/score_product

def convergence(reactant, product):
    sum = 0
    for i in reactant:
        sum = sum + abs(atom_number(product)/len(reactant) - atom_number(i))
    mae = sum/len(reactant)

    return 1/(1+mae)
