import os
import warnings
from typing import Optional, Callable, Tuple, Any, List, Iterable
import bisect

from torch.utils.data.dataset import Dataset, T_co, IterableDataset
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
import pickle as pkl
import random
from sklearn import metrics as mr
import numpy as np
import pickle as pkl
from pathlib import Path

class ImageList(datasets.VisionDataset):
    """A generic Dataset class for image classification

    Args:
        root (str): Root directory of dataset
        classes (list[str]): The names of all the classes
        data_list_file (str): File to read the image list from.
        is_train (bool): Sample multiple views if it is training mode. 
        method (str): For compatibility with different methods. 
        transform (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line has 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride :meth:`~ImageList.parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], data_list_file: str, is_train: bool, method='', sample='', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self.parse_data_file(data_list_file)
        self.targets = [s[1] for s in self.samples]
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.loader = default_loader
        self.data_list_file = data_list_file
        self.is_train = is_train
        self.method = method
        self.sample = sample

        if 'miplov2' in root or 'miplov3' in root:
            self.loader = self.cache_loader

            project_root = Path(__file__).resolve().parents[1]
            folder_name = 'dnd_cka_miplov{}'.format('2' if 'miplov2' in root else '3')

            # metadata
            prepro_metadata_path = os.path.join(project_root, 'data', folder_name, 'prepro_metadata.pkl')  
            with open(prepro_metadata_path, 'rb') as f:
                self.prepro_metadata = pkl.load(f)
            
            folder_name = 'dnd_cka_miplov{}'.format('3' if 'miplov2' in root else '2')
            prepro_metadata_path = os.path.join(project_root, 'data', folder_name, 'prepro_metadata.pkl')  
            with open(prepro_metadata_path, 'rb') as f:
                tmpt = pkl.load(f)
            # self.prepro_metadata = self.prepro_metadata | tmpt    # python>3.9
            self.prepro_metadata = {**self.prepro_metadata, **tmpt} # python>3.5

            print("Loading pre-processed metadata: ".ljust(40), prepro_metadata_path, ': ', len(self.prepro_metadata)-1) # -1 cuz prepro_metadata['dates']

            self.possible_views = {}
            self.pair_dist = {}
            
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
            return (tuple): (image, target) where target is index of the target class.
        """

        # single source & single target
        if self.method != 'mv-match' or (not self.is_train):
            path, target = self.samples[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None and target is not None:
                target = self.target_transform(target)
            return img, target
        
        # multiple views
        else:
            path, target = self.samples[index]
            img = self.loader(path)

            # 1. create subsample dict: a pool of most dissimilar views for each query
            n_possible = 5
            n_view = 1 # default=1, #views for each query image, TODO: support n_view > 1
            img_name = path.split('/')[-1][:-4]+'.jpg'
            if not self.possible_views.get(img_name): 
                multi_view_names = [key for key, value in self.prepro_metadata.items() if (type(value)==dict and value.get('date') and value['date']==self.prepro_metadata[img_name]['date'] and value['idx']==self.prepro_metadata[img_name]['idx'] and value['genotype']==self.prepro_metadata[img_name]['genotype'] and key!=img_name)]

                if self.sample == 'mutual':
                    # compute nmi list from scratch
                    target_size = 224 # for computing image-level similarity
                    im1 = self.loader(path.replace('/1344/', '/{}/'.format(target_size)))
                    im1 = np.array(im1)
                    im1 = im1[:, :, ::-1]
                    nmi_list = []
                    for im2_name in multi_view_names:
                        combined_name = img_name+im2_name if img_name<im2_name else im2_name+img_name

                        if not self.pair_dist.get(combined_name):
                            path2 = path.replace(img_name[:-4], im2_name[:-4])
                            im2 = self.loader(path2.replace('/1344/', '/{}/'.format(target_size)))
                            im2 = np.array(im2)
                            im2 = im2[:, :, ::-1].copy()

                            # nmi: smaller = more dissimilar [nmi: normalized mutual information] 
                            self.pair_dist[combined_name] = mr.normalized_mutual_info_score(im1.reshape(-1), im2.reshape(-1))       
                        
                        nmi = self.pair_dist[combined_name]
                        nmi_list.append(nmi)
                        # print(img_name, im2_name, nmi)
                    
                    # sample n_possible most dissmilar views first (smallest value = most dissimilar)
                    multi_view_names = [x for _, x in sorted(zip(nmi_list, multi_view_names))] 
                    self.possible_views[img_name] = multi_view_names[:n_possible]

                elif self.sample == 'random':
                    self.possible_views[img_name] = random.sample(multi_view_names, min(n_possible, len(multi_view_names)))
                else:
                    raise NotImplementedError
            multi_view_names = self.possible_views[img_name]
            # print(len(multi_view_names))

            # 2. sample = random
            sampled_views = random.sample(multi_view_names, n_view)

            img2_name = sampled_views[0]
            img2 = self.loader(path.replace(img_name[:-4], img2_name[:-4]))
            if self.transform is not None:
                img = self.transform(img)
                img2 = self.transform(img2)
            if self.target_transform is not None and target is not None:
                target = self.target_transform(target)
                
            # TODO: support n_view > 1
            # img3_name = sampled_views[1]
            # img3 = self.loader(path.replace(img_name[:-4], img3_name[:-4]))
            # if self.transform is not None:
            #     img3 = self.transform(img3)

            return (img, img2), target
            # return (img, img2, img3), target

    def __len__(self) -> int:
        return len(self.samples)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Args:
            file_name (str): The path of data file
            return (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    @classmethod
    def domains(cls):
        """All possible domain in this dataset"""
        raise NotImplemented
    
    def cache_loader(self, path: str) -> Any:
        # load cache files instead of raw images for acceleration
        if "/home/user" in os.getcwd():
            path = path.replace("/home/yij", "/home/user")
        with open(path, 'rb') as f:
            img = pkl.load(f)
        return img

class MultipleDomainsDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, domains: Iterable[Dataset], domain_names: Iterable[str], domain_ids) -> None:
        super(MultipleDomainsDataset, self).__init__()
        # Cannot verify that datasets is Sized
        assert len(domains) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        self.datasets = self.domains = list(domains)
        for d in self.domains:
            assert not isinstance(d, IterableDataset), "MultipleDomainsDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.domains)
        self.domain_names = domain_names
        self.domain_ids = domain_ids

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.domains[dataset_idx][sample_idx] + (self.domain_ids[dataset_idx],)

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes