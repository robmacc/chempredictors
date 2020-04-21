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


# ███╗   ██╗███████╗████████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
# ████╗  ██║██╔════╝╚══██╔══╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
# ██╔██╗ ██║█████╗     ██║   ██║ █╗ ██║██║   ██║██████╔╝█████╔╝
# ██║╚██╗██║██╔══╝     ██║   ██║███╗██║██║   ██║██╔══██╗██╔═██╗
# ██║ ╚████║███████╗   ██║   ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
# ╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
# ████████╗███████╗███████╗████████╗
# ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
#    ██║   █████╗  ███████╗   ██║
#    ██║   ██╔══╝  ╚════██║   ██║
#    ██║   ███████╗███████║   ██║
#    ╚═╝   ╚══════╝╚══════╝   ╚═╝


from .context import network


def test_constructor():
    # net = network.Network(num_features=32, learning_rate=0.01)
    net2 = network.SolubilityNetwork(num_features=32, learning_rate=0.01)
    net3 = network.OneLayerNetwork(num_features=32, learning_rate=0.01)
    assert net2.isCUDA()
    assert net3.isCUDA()


# def test_one_layer_GCN():
#     singleLayerNetwork = network.OneLayerNetwork(num_features=32,
#                                                  learning_rate=0.01)
#     singleLayerNetwork.forward(data)
#     assert singleLayerNetwork.one_layer_output ==
#     assert singleLayerNetwork.activated_output ==


# def test_forwardPass():
