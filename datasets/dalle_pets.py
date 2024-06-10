import os
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

class Dalle_Pets(DatasetBase):
    
    dataset_dir = 'dalle_oxford_pets'

    def __init__(self, cfg, clean_data=False):
        root = cfg['root_path']
        num_shots = cfg['dalle_shots']
        open_world = cfg['open_world']
        num_classes = cfg['num_classes']

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'split_dalle_pet.json')
        train = self.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, open_world=open_world, num_classes=num_classes)

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath = impath,
                    label = int(label),
                    classname = classname
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        # val = _convert(split['val'])
        # test = _convert(split['test'])

        return train