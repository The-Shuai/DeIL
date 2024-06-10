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
2. Download the ["cache"](https://drive.google.com/file/d/1cgIj_ZZUndLVbmVU-XxwySyV29fkl_I6/view?usp=drive_link), ["clipn_cache"](https://drive.google.com/file/d/1cgIj_ZZUndLVbmVU-XxwySyV29fkl_I6/view?usp=drive_link), ["gpt_file"](https://drive.google.com/file/d/1cgIj_ZZUndLVbmVU-XxwySyV29fkl_I6/view?usp=drive_link) folders, and place them in the root directory.

## Citation
```
@inproceedings{shao2024DeIL,
  title={DeIL: Direct-and-Inverse CLIP for Open-World Few-Shot Learning},
  author={Shao, Shuai and Bai, Yu and Wang, Yan and Liu, Baodi and Zhou, Yicong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
