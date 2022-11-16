from torch.utils.data import DataLoader
from data_loader.dataset_selection import DatasetSelection

class DataLoader(DataLoader):

    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.dataset = DatasetSelection(
            self.dataset_name, 
            args.split,
            remove_zero_degree_nodes=args.remove_zero_degree_nodes
        ).get_dataset()

        print("load dataset: ", self.dataset_name, "split: ", args.split_id)
