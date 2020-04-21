import rdkit.Chem
import src.molecular
import torch_geometric


class Dataset(torch_geometric.data.Dataset):
    def __init__(self, file, labels, mol_property):
        self.data = [src.molecular.molToGraph(m, mol_property, labels) for m in
                     rdkit.Chem.SDMolSupplier(file)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
