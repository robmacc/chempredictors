# Copyright 2020 Rob Maccallum

# This file is part of Objectives.

# Objectives is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Objectives is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Objectives.  If not, see <https://www.gnu.org/licenses/>.


# ███╗   ███╗ ██████╗ ██╗     ███████╗ ██████╗██╗   ██╗██╗      █████╗ ██████╗
# ████╗ ████║██╔═══██╗██║     ██╔════╝██╔════╝██║   ██║██║     ██╔══██╗██╔══██╗
# ██╔████╔██║██║   ██║██║     █████╗  ██║     ██║   ██║██║     ███████║██████╔╝
# ██║╚██╔╝██║██║   ██║██║     ██╔══╝  ██║     ██║   ██║██║     ██╔══██║██╔══██╗
# ██║ ╚═╝ ██║╚██████╔╝███████╗███████╗╚██████╗╚██████╔╝███████╗██║  ██║██║  ██║
# ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝

# adaption of methods from github.com/deepchem/deepchem/blob/master/deepchem/
# feat/graph_features.py

import rdkit.Chem
import numpy
import torch
import torch_geometric


def getIntervals(l):
    '''Parameters: l: list of lists.
    Returns: list where the elements are the cumulative products of the lengths
    of the lists in l'''

    intervals = len(l) * [0]
    intervals[0] = 1
    for k in range(1, len(l)):
        intervals[k] = (len(l[k]) + 1) * intervals[k - 1]
    return intervals


atoms_short = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
    'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
]
num_hydrogens = [0, 1, 2, 3, 4]
valences = [0, 1, 2, 3, 4, 5, 6]
formal_charges = [-3, -1, -2, 0, 1, 2, 3]
hybridizations = [
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2
]
radical_electrons = [0, 1, 2]
chiralities = ['R', 'S']
reference_lists = [
    atoms_short, num_hydrogens, valences, formal_charges, radical_electrons,
    hybridizations, chiralities
]
intervals = getIntervals(reference_lists)
atoms_long = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
    'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
    'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'
]
degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def getIndex(l, element):
    '''Parameters: l: list, element: element in l.
    Returns: the index of element in l, and len(l) if not found.'''
    try:
        return l.index(element)
    except Exception:
        return len(l)


def getFeatureList(atom):
    '''Parameters: atom: instance of rdkit.Chem.rdchem.atom
    Returns: list of features for the given atom.'''
    features = 6 * [0]
    features[0] = getIndex(atoms_short, atom.GetSymbol())
    features[1] = getIndex(num_hydrogens, atom.GetTotalNumHs())
    features[2] = getIndex(valences, atom.GetImplicitValence())
    features[3] = getIndex(formal_charges, atom.GetFormalCharge())
    features[4] = getIndex(radical_electrons, atom.GetNumRadicalElectrons())
    features[5] = getIndex(hybridizations, atom.GetHybridization())
    return features


def featuresToIndex(features_list, intervals):
    '''Parameters: features_list: list of atom features, intervals: list of
    cumulative atom features list lengths.
    Returns: features list converted to index'''
    index = 0
    for i in range(len(intervals)):
        index += features_list[i] * intervals[i]
    index = index + 1
    return index


def atomToIndex(atom):
    '''Parameters: atom: instance of rdkit.Chem.rdchem.atom
    Returns: unique index corresponding to atom type.'''
    features_list = getFeatureList(atom)
    return featuresToIndex(features_list, intervals)


def booleanOneHot(x, allowable_set):
    '''Returns: a boolean one-hot encoding of the feature x given
    the allowable_set.'''
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:"
                        .format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def atomFeatures(atom, atom_to_index=False, explicit_H=False,
                 use_chirality=False):
    '''Parameters: atom: instance of rdkit.Chem.rdchem.atom
    Returns: atom index encoding if atom_tom_index is True, atom feature
    vector otherwise.'''
    if atom_to_index:
        return numpy.array([atomToIndex(atom)])
    else:
        results = (
            booleanOneHot(atom.GetSymbol(), atoms_long) +
            booleanOneHot(atom.GetDegree(), degrees) +
            booleanOneHot(atom.GetImplicitValence(), valences) +
            [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
            booleanOneHot(atom.GetHybridization(), hybridizations) +
            [atom.GetIsAromatic()]
        )
    if not explicit_H:
        results = results + booleanOneHot(atom.GetTotalNumHs(),
                                          num_hydrogens)
    if use_chirality:
        try:
            results = results + (
                booleanOneHot(atom.GetProp('_CIPCode'), chiralities) +
                [atom.HasProp('_ChiralityPossible')]
            )
        except Exception:
            results = results + ([False, False] +
                                 [atom.HasProp('_ChiralityPossible')])
    return numpy.array(results)


def getBondPairs(mol):
    '''Parameters: mol: instance of rdkit.Chem.rdchem.Mol'''
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def bondFeatures(bond, use_chirality=False):
    '''Parameters: bond: instance of rdkit.Chem.rdchem.Bond.
    Returns: a list encoding bond features as positional booleans.'''
    bt = bond.GetBondType()
    bond_feats = [
        bt == rdkit.Chem.rdchem.BondType.SINGLE,
        bt == rdkit.Chem.rdchem.BondType.DOUBLE,
        bt == rdkit.Chem.rdchem.BondType.TRIPLE,
        bt == rdkit.Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + booleanOneHot(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
                        )
    return numpy.array(bond_feats)


def molToGraph(mol):
    '''Parameters: mol: instance of rdkit.Chem.rdchem.Mol'''
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_features = torch.tensor([atomFeatures(atom) for atom in atoms])
    edge_indices = torch.tensor(getBondPairs(mol))
    edge_attributes = torch.tensor([bondFeatures(bond) for bond in bonds])
    data = torch_geometric.data.Data(node_features, edge_indices,
                                     edge_attributes)
    return data
