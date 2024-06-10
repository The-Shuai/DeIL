import os


from .imagenet import ImageNet
from .openset import Openset


from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .dalle_imagenet import Dalle_Imagenet
from .dalle_caltech import Dalle_Caltech
from .dalle_flowers import Dalle_Flowers
from .dalle_food import Dalle_Food
from .dalle_cars import Dalle_Cars
from .dalle_dtd import Dalle_DTD
from .dalle_eurosat import Dalle_Eurosat
from .dalle_pets import Dalle_Pets
from .dalle_sun import Dalle_Sun
from .dalle_ucf import Dalle_UCF
from .dalle_fgvc import Dalle_fgvc
# from .sd_caltech import SD_Caltech

dataset_list = {
                "imagenet": ImageNet,
                "dalle_imagenet": Dalle_Imagenet,
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "SUN397": SUN397,
                "caltech-101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food-101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "dalle_caltech": Dalle_Caltech,
                "dalle_flowers": Dalle_Flowers,
                "dalle_food": Dalle_Food,
                "dalle_cars": Dalle_Cars,
                "dalle_dtd": Dalle_DTD,
                "dalle_eurosat": Dalle_Eurosat,
                "dalle_pets": Dalle_Pets,
                "dalle_sun": Dalle_Sun,
                "dalle_ucf": Dalle_UCF,
                "dalle_fgvc": Dalle_fgvc,
                # "openset": Openset,
                #######################################################

                ############################################################

                # "sd_caltech": SD_Caltech
                }


def build_dataset(dataset, cfg, clean_data=False):
# def build_dataset(cfg):
    # current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # root_path = os.path.join(current_dir,"DATA")
    # root_path = '/home/SS/datasets/CaFo_data/DATA/'
    # cfg['root_path'], cfg['shots'], cfg['open_world']
    return dataset_list[dataset](cfg, clean_data)