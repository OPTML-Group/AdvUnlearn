# Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models
###  [Project Website]() | [Arxiv Preprint]() | [Fine-tuned Weights](https://drive.google.com/drive/folders/1Nf-EJ2W3CsZwpc5blZFi7tm7o1wEiTg4?usp=sharing) | [Demo]() <br>
Our proposed robust unlearning framework, AdvUnlearn, enhances diffusion models' safety by robustly erasing unwanted concepts through adversarial training, achieving an optimal balance between concept erasure and image generation quality. 

This is the code implementation of our Robust DM Unlearning Framework: ```AdvUnlearn```, and we developed our code based on the code base of [SD](https://github.com/CompVis/stable-diffusion) and [ESD](https://github.com/rohitgandikota/erasing).


## Prepare

### Environment Setup
A suitable conda environment named ```ldm``` can be created and activated with:

```
conda env create -f environment.yaml
conda activate AdvUnlearn
```

### Files Download
* Base model - SD v1.4: download it from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt), and move it to ```models/sd-v1-4-full-ema.ckpt```
* COCO-10k (for CLIP score and FID): you can extract the image subset from COCO dataset, or you can download it from [here](https://drive.google.com/file/d/1Qgm3nNhp6ykamszN_ZvofvuzjryTsPHB/view?usp=sharing). Then, move it to `data/imgs/coco_10k`


## Code Implementation

### Step 1: AdvUnlearn [Train]

#### Hyperparameters: 
* Concept to be unlearned: `--prompt`    (e.g., 'nudity')
* Trainable module within DM: `--train_method`
* Attack generation strategy : `--attack_method`
* Number of attack steps for the adversarial prompt generation: `--attack_step`
* Adversarial prompting strategy: `--attack_type`  ('prefix_k', 'replace_k' ,'add')
* Retaining prompt dataset: `--dataset_retain`
* Utility regularization parameter: `--retain_loss_w`

#### a) Command Example: Multi-step Attack
```
python train-scripts/AdvUnlearn.py --attack_init random --attack_step 30 --retain_train 'reg' --dataset_retain 'coco_object' --prompt 'nudity' --train_method 'text_encoder_full' --retain_loss_w 0.3
```

#### b) Command Example: Fast AT variant
```
python train-scripts/AdvUnlearn.py --attack_method fast_at --attack_init random --attack_step 30 --retain_train 'reg' --dataset_retain 'coco_object' --prompt 'nudity' --train_method 'text_encoder_full'   --retain_loss_w 0.3
```

### Step 2: Attack Evaluation [Robustness Evaluation] 
Follow the instruction in [UnlearnDiffAtk](https://github.com/OPTML-Group/Diffusion-MU-Attack) to implement attacks on DMs with ```AdvUnlearn``` text encoder for robustness evaluation.



### Step 3: Image Generation Quality Evaluation [Model Utility Evaluation]
Generate 10k images for FID & CLIP evaluation 

```
bash jobs/fid_10k_generate.sh
```  

Calculate FID & CLIP scores using [T2IBenchmark](https://github.com/boomb0om/text2image-benchmark)

```
bash jobs/tri_quality_eval.sh
```   

## Checkpoints
ALL CKPTs for different DM unleanring tasks can be found [here](https://drive.google.com/drive/folders/1Nf-EJ2W3CsZwpc5blZFi7tm7o1wEiTg4?usp=sharing).
### 
| DM Unlearning Methods | Nudity  | Van Gogh  | Objects |
|:-------|:----:|:-------:| :-------:|
| [ESD](https://github.com/rohitgandikota/erasing) (Erased Stable Diffusion)  | ✅  | ✅   | ✅ 
| [FMN](https://github.com/SHI-Labs/Forget-Me-Not) (Forget-Me-Not)  | ✅ | ✅   | ✅ 
| [AC](https://github.com/nupurkmr9/concept-ablation) (Ablating Concepts)  | ❌ | ✅   | ❌ 
| [UCE](https://github.com/rohitgandikota/unified-concept-editing) (Unified Concept Editing)  | ✅  |  ✅  |  ❌
| [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency) (Saliency Unlearning)  | ✅  | ❌ |  ✅ 
| [SH](https://github.com/JingWu321/Scissorhands_ex) (ScissorHands)  | ✅  | ❌  | ✅ 
| [ED](https://github.com/JingWu321/EraseDiff) (EraseDiff)  | ✅  | ❌  | ✅ 
| [SPM](https://github.com/Con6924/SPM) (concept-SemiPermeable Membrane)   | ✅  | ✅   | ✅ 
| **AdvUnlearn (Ours)**  | ✅  | ✅   |  ✅ 



## Cite Our Work
The preprint can be cited as follows:
```
@article{zhang2024advunlearn,
  title={Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models},
  author={Zhang, Yimeng and Chen, Xin and Jia, Jinghan and Zhang, Yihua and Fan, Chongyu and Liu, Jiancheng and Mingyi, Hong and Ding, Ke and Liu, Sijia},
  year={2024}
}
```



