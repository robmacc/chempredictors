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


import torch_geometric
import progressbar
import urllib
from . import dataset

# --------------------------------
datasets_path = './data'
num_mols = 99
batch_size = 4
num_batches = num_mols//batch_size
test_frequency = 10
qm8_link = 'ftp://ftp.aip.org/epaps/journ_chem_phys/E-JCPSA6-143-043532/gdb8_22k_elec_spec.txt'  # noqa
qm9_link = 'https://ndownloader.figshare.com/files/3195389'
chembl_link = ''
solubility_classification_labels = {'(A) low': 0, '(B) medium': 1, '(C) high': 2}  # noqa
mol_property = 'SOL_classification'
# --------------------------------


def error(msg):
    print('Error: ' + msg)
    exit(1)


def loadData(train_file, test_file, labels, mol_property, batch_size, shuffle):
    """Load training and testing data files into iterables for feeding a
    neural network.

    :param train_file: file of molecules for training
    :type train_file: .sdf
    :param test_file: file of molecules for testing
    :type test_file: .sdf
    :param labels: map from property string to integer class
    :type labels: dictionary
    :param mol_property: target property for training
    :type mol_property: string
    :param batch_size: number of molecules to feed network on each iteration
    :type batch_size: int
    :param shuffle: whether or not to randomize molecule order on load
    :type shuffle: bool
    :return: pair of training and testing iterators
    :rtype: tuple (torch_geometric.data.DataLoader,
                   torch_geometric.data.DataLoader)
    """
    training_set = dataset.Dataset(train_file, labels, mol_property)
    testing_set = dataset.Dataset(test_file, labels, mol_property)
    # if num_workers != 0 then a collate_fn must be provided to DataLoader
    # otherwise the default lambda is use which is unpicklable on windows
    train_iterator = torch_geometric.data.DataLoader(training_set,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     drop_last=True)
    test_iterator = torch_geometric.data.DataLoader(testing_set,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    drop_last=True)

    return train_iterator, test_iterator


def widgets(epoch):
    return [progressbar.FormatLabel("Train Epoch: %2d " % (epoch)),
            progressbar.Bar('=', '[', ']'), ' ',
            progressbar.Percentage()]


# def downloadData(dataset):
#     if (dataset == 'qm8'):
#         fetch(dataset, qm8_link)
#         unpackQM8()
#     elif (dataset == 'qm9'):
#         fetch(dataset, qm9_link)
#         unpackQM9()
#     elif (dataset == 'chembl'):
#         fetch(dataset, chembl_link)
#         unpackChemBL()
#     else:
#         error('invalid dataset specified.')


def fetch(dataset, dataset_link):
    print('Downloading ' + dataset + ' ...')
    urllib.request.urlretrieve(dataset_link, datasets_path + '/' + dataset)
    print(dataset + ' download complete.')
