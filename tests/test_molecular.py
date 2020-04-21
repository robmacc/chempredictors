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
# ████████╗███████╗███████╗████████╗
# ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
#    ██║   █████╗  ███████╗   ██║
#    ██║   ██╔══╝  ╚════██║   ██║
#    ██║   ███████╗███████║   ██║
#    ╚═╝   ╚══════╝╚══════╝   ╚═╝


import rdkit.Chem
from .context import molecular, utils
import torch
import torch_geometric


def test_booleanOneHotEncoding():
    symbol = 'C'
    allowable_set = ['A', 'B', 'C', 'D', 'E', 'F']
    encoding = molecular.booleanOneHotEncoding(symbol, allowable_set)
    assert encoding == [False, False, True, False, False, False]


def test_readSDF():
    mol_iterator = rdkit.Chem.SDMolSupplier('%s/test-mol.sdf'
                                            % utils.datasets_path)
    mol = next(mol_iterator)
    # test file reads in correctly
    # atoms
    atoms = mol.GetAtoms()
    assert len(atoms) == 5  # hydrogens are implied
    assert atoms[0].GetSymbol() == 'C'
    assert atoms[1].GetSymbol() == 'C'
    assert atoms[2].GetSymbol() == 'C'
    assert atoms[3].GetSymbol() == 'C'
    assert atoms[4].GetSymbol() == 'C'
    # conformer
    conformer = mol.GetConformer()
    atom_positions = conformer.GetPositions()
    assert atom_positions[0][0] == 0.2606
    assert atom_positions[0][1] == 0.1503
    # -----------------------------------
    assert atom_positions[1][0] == 1.3000
    assert atom_positions[1][1] == 0.7500
    # -----------------------------------
    assert atom_positions[2][0] == 2.6000
    assert atom_positions[2][1] == 0.0000
    # -----------------------------------
    assert atom_positions[3][0] == 3.9000
    assert atom_positions[3][1] == 0.7500
    # -----------------------------------
    assert atom_positions[4][0] == 4.9394
    assert atom_positions[4][1] == 0.1503
    # bonds
    bonds = mol.GetBonds()
    assert len(bonds) == 4
    assert bonds[0].GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE
    assert bonds[1].GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE
    assert bonds[2].GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE
    assert bonds[3].GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE
    # properties
    assert mol.GetProp('SOL_classification') == '(A) low'
    assert mol.GetProp('SOL') == '-3.18'
    assert mol.GetProp('NAME') == 'n-pentane'
    assert mol.GetProp('smiles') == 'CCCCC'


def test_molToGraph():
    # readSDF
    mol_iterator = rdkit.Chem.SDMolSupplier('%s/test-mol.sdf'
                                            % utils.datasets_path)
    mol = next(mol_iterator)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    # encode atom to graph
    node_features_shape = (len(atoms),
                           molecular.atoms_features_length_no_chirality)
    adjacency_matrix_shape = (2, len(bonds) * 2)
    edge_features_shape = (len(bonds),
                           molecular.bond_features_length_no_chirality)
    graph = molecular.molToGraph(mol, 'SOL_classification',
                                 utils.solubility_classification_labels)
    assert isinstance(graph, torch_geometric.data.Data)
    assert isinstance(graph.x, torch.Tensor)
    assert isinstance(graph.edge_index, torch.Tensor)
    assert isinstance(graph.edge_attr, torch.Tensor)
    assert graph.x.shape == node_features_shape
    assert graph.edge_index.shape == adjacency_matrix_shape
    assert graph.edge_attr.shape == edge_features_shape
