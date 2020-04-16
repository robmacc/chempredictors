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


import torch
import progressbar
import urllib
import rdkit.Chem
import src.molecular

# --------------------------------
datasets_path = './data'
num_mols = 99
batch_size = 4
num_batches = num_mols//batch_size
test_frequency = 10
qm8_link = 'ftp://ftp.aip.org/epaps/journ_chem_phys/E-JCPSA6-143-043532/gdb8_22k_elec_spec.txt'
qm9_link = 'https://ndownloader.figshare.com/files/3195389'
chembl_link = ''
solubility_classification_dict = {'(A) low': 0, '(B) medium': 1, '(C) high': 2}
mol_property = 'SOL_classification'
# --------------------------------


def error(msg):
    print('Error: ' + msg)
    exit(1)


def loadData(train_file, test_file, labels, mol_property):
    '''Parameters: file: name of data file to load, labels: dictionary of
    parameters to train on. Returns: training and testing iterators ready for
    feeding to neural network.'''
    training_set = [src.molecular.molToGraph(m, mol_property, labels) for m in
                    rdkit.Chem.SDMolSupplier(train_file)]
    testing_set = [src.molecular.molToGraph(m, mol_property, labels) for m in
                   rdkit.Chem.SDMolSupplier(test_file)]

    train_iterator = torch.utils.data.DataLoader(training_set,
                                                 batch_size=batch_size,
                                                 shuffle=True, num_workers=2,
                                                 drop_last=True)
    test_iterator = torch.utils.data.DataLoader(testing_set,
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=2,
                                                drop_last=True)

    return train_iterator, test_iterator


def widgets(epoch):
    return [progressbar.FormatLabel("Train Epoch: %2d " % (epoch)),
            progressbar.Bar('=', '[', ']'), ' ',
            progressbar.Percentage()]


def downloadData(dataset):
    if (dataset == 'qm8'):
        fetch(dataset, qm8_link)
        unpackQM8()
    elif (dataset == 'qm9'):
        fetch(dataset, qm9_link)
        unpackQM9()
    elif (dataset == 'chembl'):
        fetch(dataset, chembl_link)
        unpackChemBL()
    else:
        error('invalid dataset specified.')


def fetch(dataset, dataset_link):
    print('Downloading ' + dataset + ' ...')
    urllib.request.urlretrieve(dataset_link, datasets_path + '/' + dataset)
    print(dataset + ' download complete.')
