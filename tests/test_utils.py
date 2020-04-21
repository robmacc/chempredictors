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


# ██╗   ██╗████████╗██╗██╗     ███████╗
# ██║   ██║╚══██╔══╝██║██║     ██╔════╝
# ██║   ██║   ██║   ██║██║     ███████╗
# ██║   ██║   ██║   ██║██║     ╚════██║
# ╚██████╔╝   ██║   ██║███████╗███████║
#  ╚═════╝    ╚═╝   ╚═╝╚══════╝╚══════╝
# ████████╗███████╗███████╗████████╗
# ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
#    ██║   █████╗  ███████╗   ██║
#    ██║   ██╔══╝  ╚════██║   ██║
#    ██║   ███████╗███████║   ██║
#    ╚═╝   ╚══════╝╚══════╝   ╚═╝


import torch
import torch_geometric
from .context import utils, molecular


def test_loadIterators():
    (train_iterator,
     test_iterator) = utils.loadData(
                          '%s/two-mols-train.sdf' % (utils.datasets_path),
                          '%s/two-mols-test.sdf' % (utils.datasets_path),
                          utils.solubility_classification_labels,
                          'SOL_classification',
                          batch_size=1,
                          shuffle=True
                          )
    # confirm number of mols in test and training sets
    train_mols = 0
    for mol in train_iterator:
        train_mols += 1
    test_mols = 0
    for mol in test_iterator:
        test_mols += 1
    assert train_mols == 2
    assert test_mols == 2


def test_loadedProperties():
    (train_iterator,
     test_iterator) = utils.loadData(
                          '%s/test-mol.sdf' % (utils.datasets_path),
                          '%s/test-mol.sdf' % (utils.datasets_path),
                          utils.solubility_classification_labels,
                          'SOL_classification',
                          batch_size=1,
                          shuffle=False
                          )
    # confirm graph encoding
    # train iterator
    train_shard = next(iter(train_iterator))
    num_atoms = 5
    num_bonds = 4
    # expected shapes of graph properties
    node_features_shape = (num_atoms,
                           molecular.atoms_features_length_no_chirality)
    adjacency_matrix_shape = (2, num_bonds * 2)
    edge_features_shape = (num_bonds,
                           molecular.bond_features_length_no_chirality)
    # check types of graph properties
    assert isinstance(train_shard, torch_geometric.data.Data)
    assert isinstance(train_shard.x, torch.Tensor)
    assert isinstance(train_shard.edge_index, torch.Tensor)
    assert isinstance(train_shard.edge_attr, torch.Tensor)
    # check shapes of graph properties
    assert train_shard.x.shape == node_features_shape
    assert train_shard.edge_index.shape == adjacency_matrix_shape
    assert train_shard.edge_attr.shape == edge_features_shape
    assert train_shard.y.shape == (1, 1)
    assert train_shard.y == 0
