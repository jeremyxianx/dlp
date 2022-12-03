import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    """ This method create a customized dataset


    Parameters
    ----------
    datasets: tensor
         Datasets
    transforms: transforms
        Transformation on data

    """
    def __init__(self, datasets, transforms=None):

        self.dataset = datasets
        self.transforms = transforms

    def __getitem__(self, index):
        """ This method returns a batch of data
                

        Parameters
        ----------
        index : array-like 
            Indices of data of interest



        Returns
        -------
        data : tensor
            The index of selected backdoor samples.

        target : tensor
            Target of selected data.
        
        index : array-like 
            Indices of data of interest

        """

        data, target = self.dataset[index]

        return data, target, index

    def __len__(self):
        return len(self.dataset)
