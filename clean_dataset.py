import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchvision_models

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
import dino.utils as utils
import itertools
import json
import numpy as np

import open_clip
from open_clip.transform import image_transform


# imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
#                         "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
#                         "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
#                         "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
#                         "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
#                         "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
#                         "box turtle", "banded gecko", "green iguana", "Carolina anole",
#                         "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
#                         "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
#                         "American alligator", "triceratops", "worm snake", "ring-necked snake",
#                         "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
#                         "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
#                         "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
#                         "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
#                         "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
#                         "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
#                         "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
#                         "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
#                         "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
#                         "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
#                         "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
#                         "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
#                         "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
#                         "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
#                         "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
#                         "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
#                         "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
#                         "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
#                         "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
#                         "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
#                         "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
#                         "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
#                         "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
#                         "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
#                         "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
#                         "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
#                         "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
#                         "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
#                         "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
#                         "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
#                         "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
#                         "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
#                         "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
#                         "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
#                         "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
#                         "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
#                         "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
#                         "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
#                         "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
#                         "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
#                         "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
#                         "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
#                         "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
#                         "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
#                         "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
#                         "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
#                         "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
#                         "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
#                         "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
#                         "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
#                         "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
#                         "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
#                         "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
#                         "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
#                         "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
#                         "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
#                         "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
#                         "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
#                         "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
#                         "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
#                         "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
#                         "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
#                         "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
#                         "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
#                         "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
#                         "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
#                         "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
#                         "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
#                         "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
#                         "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
#                         "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
#                         "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
#                         "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
#                         "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
#                         "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
#                         "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
#                         "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
#                         "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
#                         "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
#                         "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
#                         "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
#                         "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
#                         "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
#                         "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
#                         "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
#                         "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
#                         "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
#                         "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
#                         "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
#                         "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
#                         "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
#                         "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
#                         "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
#                         "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
#                         "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
#                         "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
#                         "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
#                         "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
#                         "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
#                         "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
#                         "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
#                         "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
#                         "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
#                         "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
#                         "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
#                         "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
#                         "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
#                         "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
#                         "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
#                         "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
#                         "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
#                         "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
#                         "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
#                         "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
#                         "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
#                         "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
#                         "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
#                         "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
#                         "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
#                         "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
#                         "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
#                         "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
#                         "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
#                         "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
#                         "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
#                         "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
#                         "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
#                         "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
#                         "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
#                         "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
#                         "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
#                         "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
#                         "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
#                         "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
#                         "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
#                         "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
#                         "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
#                         "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
#                         "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
#                         "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
#                         "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
#                         "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
#                         "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
#                         "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
#                         "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
#                         "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
#                         "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
#                         "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
#                         "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
#                         "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
#                         "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
#                         "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
#                         "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
#                         "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
#                         "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]


def compute_prototype(dalle_train_loader, dino_model):
    with torch.no_grad():
        train_features = []
        cache_keys = []
        cache_values = []
        for i, (images, target, ignore, impath) in enumerate(tqdm(dalle_train_loader)):
            images = images.cuda()
            image_features = dino_model(images)
            train_features.append(image_features)
            target = target.cuda()
            cache_values.append(target)
        cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = torch.cat(cache_values, dim=0)

    grouped_features = {}
    for i in range(cache_values.size(0)):
        label = cache_values[i].item()
        if label not in grouped_features:
            grouped_features[label] = []
        grouped_features[label].append(cache_keys[:, i])

    prototypes_dict = {}
    for label in grouped_features:
        features_list = grouped_features[label]
        prototypes_dict[label] = torch.stack(features_list, dim=1).mean(dim=1)   
    prototypes = torch.stack(list(prototypes_dict.values()), dim=1)  #  dim * class_num

    return prototypes

def get_dino_feas(train_loader, dino_model):
    with torch.no_grad():
        train_features = []
        cache_keys = []
        cache_values = []
        for i, (images, target, ignore, impath) in enumerate(tqdm(train_loader)):
            images = images.cuda()
            image_features = dino_model(images)
            train_features.append(image_features)
            target = target.cuda()
            cache_values.append(target)
        
        cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))          
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_values = torch.cat(cache_values, dim=0)

    return cache_keys, cache_values

def get_clip_feas(train_loader, clip_model):
    with torch.no_grad():
        train_features = []
        cache_keys = []
        cache_values = []
        impath_list = []
        for i, (images, target, ignore, impath) in enumerate(tqdm(train_loader)):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            train_features.append(image_features)
            target = target.cuda()
            cache_values.append(target)
            impath_list.append(impath)
        
        cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_values = torch.cat(cache_values, dim=0)

        combined_list = []
        for ii in range(len(impath_list)):
            combined_list.extend(impath_list[ii])

    return cache_keys, cache_values, combined_list


def generated_corrected_data(lbs, gt_lbs, impaths, cfg):

    if cfg['dataset'] in ['fgvc']:
        file_path = os.path.join(cfg['root_path'], 'fgvc_aircraft', 'data', 'split_cleaned_' + cfg['dataset'] + '.json')
    else:
        file_path = os.path.join(cfg['root_path'], cfg['dataset'], 'split_cleaned_' + cfg['dataset'] + '.json')

    split_cleaned_dataset = {} 

    for ii in range(len(lbs)):
        
        if cfg['dataset'] in ['caltech-101', 'ucf101', 'dtd', 'eurosat', 'food-101', 'stanford_cars']:
            new_path = os.path.join(impaths[ii].split('/')[-2:][0], impaths[ii].split('/')[-2:][1])
        elif cfg['dataset'] in ['SUN397']:
            if len(impaths[ii].split('/')) > 10:
                new_path = os.path.join(impaths[ii].split('/')[-4:][0], impaths[ii].split('/')[-4:][1], impaths[ii].split('/')[-4:][2], impaths[ii].split('/')[-4:][3])  
            else:
                new_path = os.path.join(impaths[ii].split('/')[-3:][0], impaths[ii].split('/')[-3:][1], impaths[ii].split('/')[-3:][2])        
        elif cfg['dataset'] in ['imagenet']:
            new_path = os.path.join(impaths[ii].split('/')[-3:][0], impaths[ii].split('/')[-3:][1], impaths[ii].split('/')[-3:][2])
        else:
            new_path = impaths[ii].split('/')[-1]
        new_data = [
                new_path, # path
                int(gt_lbs[ii]), # ground truth label
                '', # ground truth classname
                int(lbs[ii]), # cleaned label
                '', # cleaned classname
                0 # domain
            ]   
        
        # add new data to the dictionary
        if "train" in split_cleaned_dataset:
            split_cleaned_dataset["train"].append(new_data)
        else:
            split_cleaned_dataset["train"] = [new_data]

    json_str = json.dumps(split_cleaned_dataset)
    with open(file_path, "w") as f:
        f.write(json_str) 
    

def clean_model(cfg, train_data_dict, given_dataset, re_clean=True):
    # if cfg['open_world']['nlb'] == 0.0:
    #     re_clean = False

    if re_clean == True:
        print("\nCleaning training data.")
        img_feas = train_data_dict['img_feas']
        lbs = train_data_dict['lbs']
        gt_lbs = train_data_dict['gt_lbs']
        impaths = train_data_dict['impaths']
        logits_yes = train_data_dict['logits_yes'].to(torch.float32)
        probs_no = train_data_dict['probs_no']

        # select samples with incorrected labels based on CLIPN model 
        threshold = cfg['threshold']
        incorrect_indices_lst = []
        for idx in range(len(lbs)):
            if probs_no[idx][lbs[idx]] > threshold:
                incorrect_indices_lst.append(idx)
        
        # use CLIP model to clean selected samples with incorrect labels
        probs_yes = F.softmax(logits_yes, dim=1)
        # probs = torch.matmul(probs_yes, (1-probs_no)) 
        pred_lbs = torch.argmax(probs_yes, dim=1)  
        lbs[incorrect_indices_lst] = pred_lbs[incorrect_indices_lst] # cleaning

        generated_corrected_data(lbs, gt_lbs, impaths, cfg)

        dataset = build_dataset(cfg['dataset'], cfg, clean_data=True)
    else:
        dataset = given_dataset

    return dataset
