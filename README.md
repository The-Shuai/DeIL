# DeIL
This paper has been accepted by **CVPR 2024**. We are profoundly grateful for the significant insights provided by [CLIPN](https://github.com/xmed-lab/CLIPN).

## Abstract
Open-World Few-Shot Learning (OFSL) is a critical field of research, concentrating on the precise identification of target samples in environments with scarce data and unreliable labels, thus possessing substantial practical significance.
Recently, the evolution of foundation models like CLIP has revealed their strong capacity for representation, even in settings with restricted resources and data.
This development has led to a significant shift in focus, transitioning from the traditional method of “building models from scratch” to a strategy centered on “efficiently utilizing the capabilities of foundation models to extract relevant prior knowledge tailored for OFSL and apply it judiciously”.
Amidst this backdrop, we unveil the **D**ir**e**ct-and-**I**nverse C**L**IP (**DeIL**), an innovative method leveraging our proposed “Direct-and-Inverse” concept to activate CLIP-based methods for addressing OFSL.
This concept transforms conventional single-step classification into a nuanced two-stage process: initially filtering out less probable categories, followed by accurately determining the specific category of samples.
DeIL comprises two key components: a pre-trainer (frozen) for data denoising, and an adapter (tunable) for achieving precise final classification.

![图片1](https://github.com/The-Shuai/DeIL/blob/main/Figures/flowchart.png)

## Get Started
1. Create a conda environment and install dependencies.
```
pip install -r requirements.txt
```
2. Download the ["cache", "clipn_cache", "gpt_file" folders](https://drive.google.com/drive/u/0/folders/1mru7WbzqJ1XjDYGlbFQ_6kjLRU4aexvS), and place them in the root directory.
3. Download the [CLIPN pre-trained model](https://drive.google.com/drive/folders/1eNaaPaRWz0La8_qQliX30A4I7Y44yDMY) and place it in the "clipn_cache" directory.   
   e.g., "./clipn_cache/CLIPN_ATD_Repeat2_epoch_10.pt".  
   (please refer to [CLIPN](https://github.com/xmed-lab/CLIPN) for more details)
4. Follow [Download_OFSL_Datasets.md](https://github.com/The-Shuai/CO3/blob/main/Download_OFSL_Datasets.md) to download the datasets.
5. Modify the ```main_path``` in the [main.py](https://github.com/The-Shuai/DeIL/blob/main/main.py) file on line 26 to match the dataset you intend to validate.      
   e.g., set the ```main_path``` to ```main_path = "./configs/imagenet/config.yaml"```
6. Modify the ```root_path``` on the 2nd line of the ```config.yaml``` file corresponding to your dataset.    
   e.g., within the ```./configs/imagenet/config.yaml``` file, update the ```root_path``` to ```root_path: "./DATA/"```
7. Run
```
CUDA_VISIBLE_DEVICES=0 python main.py
```
   
## Citation
```
@inproceedings{shao2024DeIL,
  title={DeIL: Direct-and-Inverse CLIP for Open-World Few-Shot Learning},
  author={Shao, Shuai and Bai, Yu and Wang, Yan and Liu, Baodi and Zhou, Yicong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
