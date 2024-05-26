from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from .convertModels import savemodelDiffusers
import wandb
import pandas as pd
from tabulate import tabulate
from .prompt_dataset import *

def retain_prompt(dataset_retain):
    # Prompt Dataset to be retained

    if dataset_retain == 'imagenet243':
        retain_dataset = PromptDataset('./data/prompts/train/imagenet243_retain.csv')
    elif dataset_retain == 'imagenet243_no_filter':
        retain_dataset = PromptDataset('./data/prompts/train/imagenet243_no_filter_retain.csv')
    elif dataset_retain == 'coco_object':
        retain_dataset = PromptDataset('./data/prompts/train/coco_object_retain.csv')
    elif dataset_retain == 'coco_object_no_filter':
        retain_dataset = PromptDataset('./data/prompts/train/coco_object_no_filter_retain.csv')
    else:
        raise ValueError('Invalid dataset for retaining prompts')
    
    return retain_dataset


def param_choices(model, train_method, component='all', final_layer_norm=False):
    # choose parameters to train based on train_method
    parameters = []
    
    # Text Encoder FUll Weight Tuning
    if train_method == 'text_encoder_full':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Final Layer Norm
            if name.startswith('final_layer_norm'):
                if component == 'all' or final_layer_norm==True:
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            # Transformer layers 
            elif name.startswith('encoder'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            # Embedding layers
            else:
                pass
           
    # Text Encoder Layer 0 Tuning
    elif train_method == 'text_encoder_layer0':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer01':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer012':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer0123':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer01234':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer012345':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer0123456':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer01234567':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer012345678':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer0123456789':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8') or name.startswith('encoder.layers.9'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer012345678910':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8') or name.startswith('encoder.layers.9') or name.startswith('encoder.layers.10'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer01234567891011':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8') or name.startswith('encoder.layers.9') or name.startswith('encoder.layers.10') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer0_11':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    
    elif train_method == 'text_encoder_layer01_1011':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.10') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer012_91011':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.9') or name.startswith('encoder.layers.10') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass

    # UNet Model Tuning
    else:
        for name, param in model.model.diffusion_model.named_parameters():
            # train all layers except x-attns and time_embed layers
            if train_method == 'noxattn':
                if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                    pass
                else:
                    print(name)
                    parameters.append(param)
                    
            # train only self attention layers
            if train_method == 'selfattn':
                if 'attn1' in name:
                    print(name)
                    parameters.append(param)
                    
            # train only x attention layers
            if train_method == 'xattn':
                if 'attn2' in name:
                    print(name)
                    parameters.append(param)
                    
            # train all layers
            if train_method == 'full':
                print(name)
                parameters.append(param)
                
            # train all layers except time embed layers
            if train_method == 'notime':
                if not (name.startswith('out.') or 'time_embed' in name):
                    print(name)
                    parameters.append(param)
            if train_method == 'xlayer':
                if 'attn2' in name:
                    if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                        print(name)
                        parameters.append(param)
            if train_method == 'selflayer':
                if 'attn1' in name:
                    if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                        print(name)
                        parameters.append(param)
    
    return parameters


def str2id(tokenizer, prompt, device):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",truncation=True
    )
    return text_input.input_ids.to(device)
    
def img2latent(vae, image, device):
    with torch.no_grad():
        img_input  = image.unsqueeze(0).to(device)
        x0 = vae.encode(img_input).latent_dist.mean
        x0 *= 0.18215
    return x0
    
def id2embedding(tokenizer, all_embeddings, input_ids, device):
    input_one_hot = F.one_hot(input_ids.view(-1), num_classes = len(tokenizer.get_vocab())).float()
    input_one_hot = torch.unsqueeze(input_one_hot,0).to(device)
    input_embeds = input_one_hot @ all_embeddings
    return input_embeds


# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)

##################### ESD Functions
def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler


def save_model(folder_path, model, name, num, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    # SAVE MODEL

    # PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'
    folder_path = f'{folder_path}/models'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/Compvis-UNet-{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/Compvis-UNet-{name}.pt'
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format') 
        savemodelDiffusers(path, name, compvis_config_file, diffusers_config_file, device=device )
        
def save_text_encoder(folder_path, model, name, num):
    # SAVE MODEL

    # PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'
    folder_path = f'{folder_path}/models'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/TextEncoder-{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/TextEncoder-{name}.pt'
    
    torch.save(model.state_dict(), path)


def save_history(folder_path, losses, word_print):
    folder_path = f'{folder_path}/logs'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)
    
def print_args_table(parser, args):
    args_info = {
        'Argument': [],
        # 'Description': [],
        # 'Type': [],
        'Choices': [],
        # 'Default': [],
        # 'Required': [],
        'Selected Value': []
    }

    for action in parser._actions:
        if action.dest != 'help':  
            args_info['Argument'].append(action.option_strings[0])
            # args_info['Description'].append(action.help)
            # args_info['Type'].append(type(action.default).__name__)
            args_info['Choices'].append(', '.join(action.choices) if action.choices else 'N/A')
            # args_info['Default'].append(action.default)
            # args_info['Required'].append(action.required)
            args_info['Selected Value'].append(getattr(args, action.dest))
            
    args_df = pd.DataFrame(args_info)

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)

    print(tabulate(args_df, headers='keys', tablefmt='grid', colalign=("left",), maxcolwidths=[20, 20, 20]))