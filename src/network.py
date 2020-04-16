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


import torch
import torch_geometric
import progressbar
import src.utils


def setDevice(constructor):
    def wrappedConstructor(constructor):
        network = constructor()
        # network.to(network.device)
        return network
    return wrappedConstructor


class Network(torch.nn.Module):
    # @setDevice
    def __init__(self, num_features):
        super(Network, self).__init__()
        self.graph_conv1 = torch_geometric.nn.GCNConv(num_features, 128,
                                                      cached=False)
        self.batch_norm1 = torch.nn.BatchNorm1d(128)
        self.graph_conv2 = torch_geometric.nn.GCNConv(128, 64, cached=False)
        self.batch_norm2 = torch.nn.BatchNorm1d(64)
        self.fully_connected1 = torch.nn.Linear(64, 64)
        self.batch_norm3 = torch.nn.BatchNorm1d(64)
        self.fully_connected2 = torch.nn.Linear(64, 64)
        self.fully_connected3 = torch.nn.Linear(64, 3)

        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'cpu')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = torch.nn.functional.relu(self.graph_conv1(x, edge_index))
        x = self.batch_norm1(x)
        x = torch.nn.functional.relu(self.graph_conv2(x, edge_index))
        x = self.batch_norm2(x)
        x = torch_geometric.nn.global_add_pool(x, batch.batch)
        x = torch.nn.functional.relu(self.fully_connected1(x))
        x = self.batch_norm3(x)
        x = torch.nn.functional.relu(self.fully_connected2(x))
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.fully_connected3(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

    def propagate(self, train_iterator, progress_bar):
        for batch_num, batch in enumerate(train_iterator):
            output = self.forward(batch)
            loss = torch.nn.functional.nll_loss(output, batch.y)
            loss.backward()
            self.optimizer.step()
            progress_bar.update(batch_num)

    def initProgressBar(self, epoch):
        progress_bar = progressbar.ProgressBar(0, src.utils.num_batches,
                                               src.utils.widgets(epoch))
        progress_bar.start()
        return progress_bar

    def test(self, test_iterator):
        test_loss = 0
        for batch in test_iterator:
            output = self.forward(batch)
            batch_loss = torch.nn.functional.nll_loss(output, batch.y)
            test_loss += batch_loss
        src.utils.printTestLoss(test_loss)

    def train(self, train_iterator, test_iterator, epochs):
        self.optimizer.zero_grad()
        for epoch in range(epochs):
            progress_bar = self.initProgressBar(epoch)
            self.propagate(train_iterator, progress_bar)
            if epoch % src.utils.test_frequency == 0:
                self.test(test_iterator)
