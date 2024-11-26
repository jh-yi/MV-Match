from typing import Optional
import os
from .imagelist import ImageList

class MiPlo(ImageList):

    image_list = {
        "B": "B",
        "H": "H",
        "M": "M", 
        "J": "J", 
    }
    CLASSES = ['-N', '-P', '-K', '-B', '-S', 'control']

    def __init__(self, root: str, task: str, split: Optional[str] = 'train', download: Optional[bool] = False, **kwargs):
        assert task in self.image_list
        assert split in ['train', 'test']

        list_name = "image_list"
        data_list_file = os.path.join(root, list_name, "{}_{}.txt".format(self.image_list[task], split))
        print("loading {}".format(data_list_file))

        super(MiPlo, self).__init__(root, MiPlo.CLASSES, data_list_file=data_list_file, **kwargs)
    
    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())