import os
import math
import random
from collections import defaultdict

import torchvision.transforms as transforms

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .dalle_imagenet import Dalle_Imagenet


imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
                        "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
                        "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
                        "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
                        "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
                        "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
                        "box turtle", "banded gecko", "green iguana", "Carolina anole",
                        "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
                        "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
                        "American alligator", "triceratops", "worm snake", "ring-necked snake",
                        "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
                        "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
                        "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
                        "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
                        "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
                        "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
                        "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
                        "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
                        "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
                        "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
                        "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
                        "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
                        "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
                        "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
                        "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
                        "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
                        "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
                        "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
                        "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
                        "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
                        "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
                        "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
                        "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
                        "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
                        "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
                        "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
                        "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
                        "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
                        "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
                        "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
                        "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
                        "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
                        "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
                        "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
                        "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
                        "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
                        "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
                        "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
                        "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
                        "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
                        "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
                        "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
                        "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
                        "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
                        "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
                        "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
                        "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
                        "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
                        "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
                        "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
                        "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
                        "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
                        "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
                        "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
                        "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
                        "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
                        "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
                        "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
                        "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
                        "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
                        "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
                        "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
                        "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
                        "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
                        "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
                        "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
                        "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
                        "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
                        "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
                        "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
                        "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
                        "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
                        "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
                        "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
                        "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
                        "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
                        "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
                        "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
                        "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
                        "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
                        "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
                        "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
                        "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
                        "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
                        "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
                        "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
                        "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
                        "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
                        "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
                        "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
                        "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
                        "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
                        "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
                        "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
                        "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
                        "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
                        "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
                        "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
                        "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
                        "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
                        "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
                        "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
                        "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
                        "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
                        "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
                        "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
                        "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
                        "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
                        "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
                        "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
                        "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
                        "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
                        "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
                        "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
                        "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
                        "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
                        "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
                        "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
                        "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
                        "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
                        "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
                        "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
                        "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
                        "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
                        "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
                        "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
                        "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
                        "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
                        "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
                        "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
                        "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
                        "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
                        "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
                        "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
                        "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
                        "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
                        "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
                        "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
                        "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
                        "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
                        "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
                        "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
                        "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
                        "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
                        "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
                        "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
                        "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
                        "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
                        "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
                        "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
                        "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
                        "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
                        "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
                        "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
                        "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]


template = ['a photo of a {}, a type of pet.']


class ImageNet(DatasetBase):

    dataset_dir = 'imagenet'
    # dataset_ood_dir = 'open_set' # ood: out-of-distribution

    def __init__(self, cfg, clean_data=False):
        root = cfg['root_path']
        num_shots = cfg['shots']
        open_world = cfg['open_world']
        num_classes = cfg['num_classes']

        # self.dataset_ood_dir = os.path.join(root, self.dataset_ood_dir)
        # self.image_ood_dir = os.path.join(self.dataset_ood_dir, 'images')
        # self.split_ood_path = os.path.join(self.dataset_ood_dir, 'split_open_data.json')

        self.template = template

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        if clean_data == False:
            self.split_path = os.path.join(self.dataset_dir, 'split_imagenet.json')
            train, val = self.read_split(self.split_path, self.image_dir)
            train = self.generate_fewshot_dataset(train, num_shots=num_shots) 
            test = val
        else:
            self.split_path = os.path.join(self.dataset_dir, 'split_cleaned_imagenet.json')
            train = self.read_split_(self.split_path, self.image_dir)
            val = None
            test = None 
        
        # if cfg['open_world']['is_open_world'] == True:
        #     # train_ds, val_ds = self.read_split(self.split_ds_path, self.image_ds_dir) # ds: domain shift
        #     train_ood, val_ood, test_ood = self.read_split_(self.split_ood_path, self.image_ood_dir) # ood: out-of-distribution         
        #     val = self.generate_open_world_test_data(cfg, val, val_ood) 
        #     test = self.generate_open_world_test_data(cfg, test, test_ood) 

        super().__init__(train_x=train, val=val, test=test, open_world=open_world, num_classes=num_classes, clean_data=clean_data)
    
    def read_data(self,root, split_file):
        filepath = os.path.join(root,self.anno_dir, split_file)
        items = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(' ')
                breed = imname.split('_')[:-1]
                breed = '_'.join(breed)
                breed = breed.lower()
                imname += '.jpg'
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1 # convert to 0-based index
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=breed
                )
                items.append(item)
        
        return items
    
    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f'Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val')
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)
        
        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)
        
        return train, val
    
    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, '')
                if impath.startswith('/'):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out
        
        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {
            'train': train,
            'val': val
            # 'test': test
        }

        write_json(split, filepath)
        print(f'Saved split to {filepath}')
    
    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname, incorrect_label, incorrect_classname, domain in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath = impath,
                    label = int(label),
                    classname = classname,
                    incorrect_label = int(incorrect_label),
                    incorrect_classname = incorrect_classname,
                    domain = domain
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        # test = _convert(split['test'])

        return train, val
    
    # @staticmethod
    # def read_split_(filepath, path_prefix):
    #     def _convert(items):
    #         out = []
    #         for impath, label, classname, domain in items:
    #             impath = os.path.join(path_prefix, impath)
    #             item = Datum(
    #                 impath = impath,
    #                 label = int(label),
    #                 classname = classname,
    #                 domain = domain
    #             )
    #             out.append(item)
    #         return out
        
    #     print(f'Reading split from {filepath}')
    #     split = read_json(filepath)
    #     train = _convert(split['train'])
    #     val = _convert(split['val'])
    #     test = _convert(split['test'])

    #     return train, val, test
    
    @staticmethod
    def read_split_(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, gt_lb, classname, cleaned_lb, cleaned_classname, domain in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath = impath,
                    label = int(gt_lb),
                    classname = classname,
                    incorrect_label = int(cleaned_lb),
                    incorrect_classname = cleaned_classname
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        # val = _convert(split['val'])
        # test = _convert(split['test'])

        return train