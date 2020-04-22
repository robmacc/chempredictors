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
bonds = [rdkit.Chem.rdchem.BondType.SINGLE,
         rdkit.Chem.rdchem.BondType.DOUBLE,
         rdkit.Chem.rdchem.BondType.TRIPLE,
         rdkit.Chem.rdchem.BondType.AROMATIC]
isomers = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
atoms_features_length_no_chirality = (len(atoms_long) + len(degrees)
                                      + len(valences) + 2
                                      + len(hybridizations)
                                      + 1 + len(num_hydrogens))
bond_features_length_no_chirality = (len(bonds) + 1 + 1)


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
    ''':param atom: instance of :class:`rdkit.Chem.rdchem.Atom`
    :return: unique index corresponding to atom type'''
    features_list = getFeatureList(atom)
    return featuresToIndex(features_list, intervals)


def booleanOneHotEncoding(x, allowable_set):
    '''Returns: a boolean one-hot encoding of the feature x given
    the allowable_set.'''
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set {1}:"
                        .format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def encodeAtomFeatures(atom, atom_to_index=False, explicit_H=False,
                       use_chirality=False):
    """Encode the given atom into it's node feature vector.

    :param atom: atom instance
    :type atom: rdkit.Chem.rdchem.atom
    :param atom_to_index: whether to use index representation, defaults to
                          False
    :type atom_to_index: bool, optional
    :param explicit_H: whether hydrogens are treated as explicit atoms or
                       inferred, defaults to False
    :type explicit_H: bool, optional
    :param use_chirality: whether to include the molecules chirality in node
                          feature vectors, defaults to False
    :type use_chirality: bool, optional
    :return: node feature vector
    :rtype: numpy.array
    """
    if atom_to_index:
        return numpy.array([atomToIndex(atom)])
    else:
        results = (
            booleanOneHotEncoding(atom.GetSymbol(), atoms_long) +
            booleanOneHotEncoding(atom.GetDegree(), degrees) +
            booleanOneHotEncoding(atom.GetImplicitValence(), valences) +
            [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
            booleanOneHotEncoding(atom.GetHybridization(), hybridizations) +
            [atom.GetIsAromatic()]
        )
    if not explicit_H:
        results = results + booleanOneHotEncoding(atom.GetTotalNumHs(),
                                                  num_hydrogens)
    if use_chirality:
        try:
            results = results + (
                booleanOneHotEncoding(atom.GetProp('_CIPCode'), chiralities) +
                [atom.HasProp('_ChiralityPossible')]
            )
        except Exception:
            results = results + ([False, False] +
                                 [atom.HasProp('_ChiralityPossible')])
    return numpy.array(results)

    # '''Parameters: atom: instance of rdkit.Chem.rdchem.atom
    # Returns: atom index encoding if atom_tom_index is True, atom feature
    # vector otherwise.'''


def generateAdjacencyMatrix(mol):
    """Generates adjacency matrix for the given molecule.

    :param mol: molecule
    :type mol: :class:`rdkit.Chem.rdchem.Mol`
    :return: adjacency matrix
    :rtype: list
    """
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def encodeBondFeatures(bond, use_chirality=False):
    """Encode the given bond into it's edge feature vector.

    :param bond: molecular bond
    :type bond: :class:`rdkit.Chem.rdchem.Bond`
    :param use_chirality: whether to include bond chirality in the encoding,
                          defaults to False
    :type use_chirality: bool, optional
    :return: encoded bond
    :rtype: `numpy.array`
    """
    bond_feats = booleanOneHotEncoding(bond.GetBondType(), bonds)
    bond_feats += [bond.GetIsConjugated()]
    bond_feats += [bond.IsInRing()]
    if use_chirality:
        bond_feats = bond_feats + booleanOneHotEncoding(str(bond.GetStereo()),
                                                        isomers)
    return numpy.array(bond_feats)


def molToGraph(mol, mol_property, labels):
    """Converts a rdkit molecule to a torch geometric graph.

    :param mol: molecule instance
    :type mol: :class:`rdkit.Chem.rdchem.Mol`
    :param mol_property: target physical or chemical property
    :type mol_property: string
    :param labels: dictionary of integers corresponding to property values
    :type labels: dictionary
    :return: torch geometric graph
    :rtype: :class:`torch_geometric.data.Data`
    """
    label = torch.tensor([[labels[mol.GetProp(mol_property)]]])
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_features = torch.tensor([encodeAtomFeatures(atom) for atom in atoms])
    edge_indices = torch.tensor(generateAdjacencyMatrix(mol))
    edge_attributes = torch.tensor([encodeBondFeatures(bond) for bond in bonds])  # noqa
    data_shard = torch_geometric.data.Data(x=node_features,
                                           edge_index=edge_indices,
                                           edge_attr=edge_attributes,
                                           y=label)
    return data_shard
