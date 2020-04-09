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
import glob
import numpy
import progressbar

# --------------------------------
dataset_path = './data'
num_mols = 99
batch_size = 4
num_batches = num_mols//batch_size
test_frequency = 10
# --------------------------------


def error(msg):
    print('Error: ' + msg)
    exit(1)


def molToArray():


def loadData():
    # read data from disk
    mol_list = glob.glob(dataset_path)
    expected_num = num_mols
    x = numpy.array([molToArray(mol) for mol in mol_list])
    if len(x) != expected_num:
        error('Expected to find %d mols' % expected_num)

    # train-test split
    train_mols = x[0:round(0.8*expected_num)]  # ~80%
    test_mols = x[round(0.8*expected_num):]  # ~20%

    # make iterators
    train_iterator = torch.utils.data.DataLoader(train_mols,
                                                 batch_size=batch_size,
                                                 shuffle=True, num_workers=2)
    test_iterator = torch.utils.data.DataLoader(test_mols,
                                                batch_size=batch_size,
                                                shuffle=False, num_workers=2)

    return train_iterator, test_iterator


def widgets(epoch):
    return [progressbar.FormatLabel("Train Epoch: %2d " % (epoch)),
            progressbar.Bar('=', '[', ']'), ' ',
            progressbar.Percentage()]
