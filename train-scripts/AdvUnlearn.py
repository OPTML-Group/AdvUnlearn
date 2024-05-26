from utils.util import *
from utils.attack_util import *
from utils.text_encoder import CustomTextEncoder
from utils.get_loss import *
from utils.prompt_dataset import *

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
import torch
from tqdm import tqdm
import random
import argparse
import wandb
from pathlib import Path
import os


def AdvUnlearn(prompt, dataset_retain, retain_batch, retain_train, retain_step, retain_loss_w , attack_method, train_method, norm_layer, component, start_guidance, negative_guidance, iterations, save_interval, lr, config_path, ckpt_path, diffusers_config_path, output_dir, devices, seperator=None, image_size=512, ddim_steps=50, adv_prompt_num=3, attack_embd_type='word_embd', attack_type='prefix_k', attack_init='latest', warmup_iter=200, attack_step=30, attack_lr=1e-2, adv_prompt_update_step=20):
    
    quick_sample_till_t = lambda x, s, code, batch, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, n_samples=batch, till_T=t, verbose=False)
    
    # ====== Stage 0: PROMPT CLEANING ======
    word_print = prompt.replace(' ','')
    if prompt == 'allartist':
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    print(f'The Concept Prompt to be unlearned:{words}')
    
    retain_dataset = retain_prompt(dataset_retain)
    
    # ======= Stage 1: TRAINING SETUP =======
    ddim_eta = 0
    model_name_or_path = "CompVis/stable-diffusion-v1-4"
    cache_path = ".cache"
    vae = AutoencoderKL.from_pretrained(model_name_or_path, subfolder="vae", cache_dir=cache_path).to(devices[0])  
    tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path, subfolder="tokenizer", cache_dir=cache_path)
    text_encoder = CLIPTextModel.from_pretrained(model_name_or_path, subfolder="text_encoder", cache_dir=cache_path).to(devices[0])
    custom_text_encoder = CustomTextEncoder(text_encoder).to(devices[0])
    all_embeddings = custom_text_encoder.get_all_embedding().unsqueeze(0)
    
    model_orig, sampler_orig, model, sampler = get_models(config_path, ckpt_path, devices)
    model_orig.eval()

    # Setup tainable model parameters
    if 'text_encoder' in train_method:
        parameters = param_choices(model=custom_text_encoder, train_method=train_method, component=component, final_layer_norm=norm_layer)
    else:
        parameters = param_choices(model=model, train_method=train_method, component=component, final_layer_norm=norm_layer)
    
    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    name = train_method
    
    # ========== Stage 2: Training ==========
    pbar = tqdm(range(iterations))
    global_step = 0
    attack_round = 0
    for i in pbar:
        # Change unlearned concept and obtain its corresponding adv embedding
        if i % adv_prompt_update_step == 0:
            
            # Reset the dataset if all prompts are seen           
            if retain_dataset.check_unseen_prompt_count() < retain_batch:
                retain_dataset.reset()
            
            word = random.sample(words,1)[0]
            text_input = tokenizer(
                word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",truncation=True
            )
            text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])
            
            # get conditional embedding for the prompt
            emb_0 = model_orig.get_learned_conditioning([''])
            emb_p = model_orig.get_learned_conditioning([word])
            
            # ===== ** Attack ** : get adversarial prompt
            if i >= warmup_iter:
                custom_text_encoder.text_encoder.eval()
                custom_text_encoder.text_encoder.requires_grad_(False)
                model.eval()
                if attack_round == 0:
                    if attack_embd_type == 'word_embd':
                        adv_word_embd, adv_input_ids = soft_prompt_attack(global_step, word, model, model_orig, tokenizer, custom_text_encoder, sampler, emb_0, emb_p,  start_guidance,  devices, ddim_steps, ddim_eta, image_size, criteria, adv_prompt_num, all_embeddings, attack_round, attack_type,  attack_embd_type, attack_step, attack_lr, attack_init, None, attack_method)
                    elif attack_embd_type == 'condition_embd':
                        adv_condition_embd, adv_input_ids = soft_prompt_attack(global_step, word, model, model_orig, tokenizer, custom_text_encoder, sampler, emb_0, emb_p, start_guidance,  devices, ddim_steps, ddim_eta, image_size, criteria, adv_prompt_num, all_embeddings, attack_round, attack_type, attack_embd_type, attack_step, attack_lr, attack_init, None, attack_method) 
                else:
                    if attack_embd_type == 'word_embd':
                        adv_word_embd, adv_input_ids = soft_prompt_attack(global_step, word, model, model_orig, tokenizer, custom_text_encoder, sampler, emb_0, emb_p,  start_guidance,  devices, ddim_steps, ddim_eta, image_size, criteria, adv_prompt_num, all_embeddings, attack_round, attack_type,  attack_embd_type, attack_step, attack_lr, attack_init, adv_word_embd, attack_method)
                    elif attack_embd_type == 'condition_embd':
                        adv_condition_embd, adv_input_ids = soft_prompt_attack(global_step, word, model, model_orig, tokenizer, custom_text_encoder, sampler, emb_0, emb_p, start_guidance,  devices, ddim_steps, ddim_eta, image_size, criteria, adv_prompt_num, all_embeddings, attack_round, attack_type, attack_embd_type, attack_step, attack_lr, attack_init, adv_condition_embd, attack_method) 
                
                global_step += attack_step
                attack_round += 1
                
        
        # Set model/TextEnocder to train or eval mode
        if 'text_encoder' in train_method:
            custom_text_encoder.text_encoder.train()
            custom_text_encoder.text_encoder.requires_grad_(True)
            model.eval()
            # print('==== Train text_encoder ====')
        else:
            custom_text_encoder.text_encoder.eval()
            custom_text_encoder.text_encoder.requires_grad_(False)
            model.train()
        opt.zero_grad()
        
        # Retaining prompts for retaining regularized training
        if retain_train == 'reg':
            retain_words = retain_dataset.get_random_prompts(retain_batch)
            retain_text_input = tokenizer(
                retain_words, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",truncation=True
            )
            retain_input_ids = retain_text_input.input_ids.to(devices[0])
            
            # retain_emb_0 = model_orig.get_learned_conditioning(['']*retain_batch)
            retain_emb_p = model_orig.get_learned_conditioning(retain_words)
            
            retain_text_embeddings = id2embedding(tokenizer, all_embeddings, retain_text_input.input_ids.to(devices[0]), devices[0])
            retain_text_embeddings = retain_text_embeddings.reshape(retain_batch, -1, retain_text_embeddings.shape[-1])  # [batch, 77, 768]
            retain_emb_n = custom_text_encoder(input_ids = retain_input_ids, inputs_embeds=retain_text_embeddings)[0]
        else:
            retain_text_input = None
            retain_text_embeddings = None
            # retain_emb_0 = None
            retain_emb_p = None
            retain_emb_n = None
        
        if i < warmup_iter:
            # Warmup training
            input_ids = text_input.input_ids.to(devices[0])
            emb_n = custom_text_encoder(input_ids = input_ids, inputs_embeds=text_embeddings)[0]
            loss = get_train_loss_retain(retain_batch, retain_train, retain_loss_w, model, model_orig, custom_text_encoder, sampler, emb_0, emb_p, retain_emb_p, emb_n, retain_emb_n, start_guidance, negative_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, input_ids, attack_embd_type)
        else:
            if attack_embd_type == 'word_embd':
                loss = get_train_loss_retain(retain_batch, retain_train, retain_loss_w, model, model_orig, custom_text_encoder, sampler, emb_0, emb_p, retain_emb_p, None, retain_emb_n, start_guidance, negative_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, adv_input_ids, attack_embd_type, adv_word_embd)
            elif attack_embd_type == 'condition_embd':
                loss = get_train_loss_retain(retain_batch, retain_train, retain_loss_w, model, model_orig, custom_text_encoder, sampler, emb_0, emb_p, retain_emb_p, None, retain_emb_n, start_guidance, negative_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, adv_input_ids, attack_embd_type, adv_condition_embd)
        
        # update weights to erase the concept
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        wandb.log({'Train_Loss':loss.item()}, step=global_step)
        wandb.log({'Attack_Loss': 0.0}, step=global_step)
        global_step += 1
        
        opt.step()
        
        if retain_train == 'iter':
            for r in range(retain_step):
                print(f'==== Retain Training at step {r} ====')
                opt.zero_grad()
                if retain_dataset.check_unseen_prompt_count() < retain_batch:
                    retain_dataset.reset()
                retain_words = retain_dataset.get_random_prompts(retain_batch)
                
                t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
                # time step from 1000 to 0 (0 being good)
                og_num = round((int(t_enc)/ddim_steps)*1000)
                og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
                t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
                retain_start_code = torch.randn((retain_batch, 4, 64, 64)).to(devices[0])
                
                retain_emb_p = model_orig.get_learned_conditioning(retain_words)
                retain_z = quick_sample_till_t(retain_emb_p.to(devices[0]), start_guidance, retain_start_code, retain_batch, int(t_enc)) # emb_p seems to work better instead of emb_0
                retain_e_p = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_p.to(devices[0]))
                
                retain_text_input = tokenizer(
                    retain_words, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",truncation=True
                )
                retain_input_ids = retain_text_input.input_ids.to(devices[0])
                retain_text_embeddings = id2embedding(tokenizer, all_embeddings, retain_text_input.input_ids.to(devices[0]), devices[0])
                retain_text_embeddings = retain_text_embeddings.reshape(retain_batch, -1, retain_text_embeddings.shape[-1])  # [batch, 77, 768]
                retain_emb_n = custom_text_encoder(input_ids = retain_input_ids, inputs_embeds=retain_text_embeddings)[0]
                retain_e_n = model.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_n.to(devices[0]))
                
                retain_loss = criteria(retain_e_n.to(devices[0]), retain_e_p.to(devices[0]))
                retain_loss.backward()
                opt.step()
        
        # ====== Stage 3: save final model and loss curve ======
        # save checkpoint and loss curve
        if (i+1) % save_interval == 0 and i+1 != iterations and i+1>= save_interval:
            if 'text_encoder' in train_method:
                save_text_encoder(output_dir, custom_text_encoder, name, i)
            else:
                save_model(output_dir, model, name, i, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

        if i % 1 == 0:
            save_history(output_dir, losses, word_print)

    # Save final model and loss curve
    model.eval()
    custom_text_encoder.text_encoder.eval()
    custom_text_encoder.text_encoder.requires_grad_(False)
    if 'text_encoder' in train_method:
        save_text_encoder(output_dir, custom_text_encoder, name, i)
    else:
        save_model(output_dir, model, name, i, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    save_history(output_dir, losses, word_print)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'AdvUnlearn',
                    description = 'Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models')
    
    # Diffusion setup
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    
    # Training setup
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=False, default='nudity')
    parser.add_argument('--dataset_retain', help='prompts corresponding to non-target concept to retain', type=str, required=False, default='coco', choices=['coco_object', 'coco_object_no_filter', 'imagenet243', 'imagenet243_no_filter'])
    # parser.add_argument('--unlearn_batch', help='batch size of unlearning prompt during training', type=int, required=False, default=5)
    parser.add_argument('--retain_batch', help='batch size of retaining prompt during training', type=int, required=False, default=5)
    parser.add_argument('--retain_train', help='different retaining version: reg (regularization) or iter (iterative)', type=str, required=False, default='iter', choices=['iter', 'reg'])
    parser.add_argument('--retain_step', help='number of steps for retaining prompts', type=int, required=False, default=1)
    parser.add_argument('--retain_loss_w', help='retaining loss weight', type=float, required=False, default=1.0)
    
    parser.add_argument('--train_method', help='method of training', type=str, choices=['text_encoder_full', 'text_encoder_layer0', 'text_encoder_layer01', 'text_encoder_layer012', 'text_encoder_layer0123', 'text_encoder_layer01234', 'text_encoder_layer012345', 'text_encoder_layer0123456', 'text_encoder_layer01234567', 'text_encoder_layer012345678', 'text_encoder_layer0123456789', 'text_encoder_layer012345678910', 'text_encoder_layer01234567891011', 'text_encoder_layer0_11','text_encoder_layer01_1011', 'text_encoder_layer012_91011', 'noxattn', 'selfattn', 'xattn', 'full', 'notime', 'xlayer', 'selflayer'], default='text_encoder_full', required=False)
    parser.add_argument('--norm_layer', help='During training, norm layer to be updated or not', action='store_true', default=False, required=False)
    parser.add_argument('--attack_method', help='method of training', type=str, choices=['pgd', 'multi_pgd', 'fast_at', 'free_at'], default='pgd', required=False)
    parser.add_argument('--component', help='component', type=str, choices=['all', 'ffn', 'attn'], default='all', required=False)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--save_interval', help='iterations used to train', type=int, required=False, default=200)
    parser.add_argument('--lr', help='learning rate used to train', type=int, required=False, default=1e-5)
    
    # Attack hyperparameters
    parser.add_argument('--adv_prompt_num', help='number of prompt token for adversarial soft prompt learning', type=int, required=False, default=1)
    parser.add_argument('--attack_embd_type', help='the adversarial embd type: word embedding, condition embedding', type=str, required=False, default='word_embd', choices=['word_embd', 'condition_embd'] )
    parser.add_argument('--attack_type', help='the attack type: append or add', type=str, required=False, default='prefix_k', choices=['replace_k' ,'add', 'prefix_k', 'suffix_k', 'mid_k', 'insert_k', 'per_k_words'])
    parser.add_argument('--attack_init', help='the attack init: random or latest', type=str, required=False, default='latest', choices=['random', 'latest'])
    parser.add_argument('--attack_step', help='adversarial attack steps', type=int, required=False, default=30)
    parser.add_argument('--adv_prompt_update_step', help='after every n step, adv prompt would be updated', type=int, required=False, default=1)
    parser.add_argument('--attack_lr', help='learning rate used to train', type=float, required=False, default=1e-3)
    parser.add_argument('--warmup_iter', help='the number of warmup interations before attack', type=int, required=False, default=200)
    
    # Log details
    parser.add_argument('--project_name', help='wandb project name', type=str, required=False, default='AdvUnlearn')
    
    args = parser.parse_args()
    
    prompt = args.prompt  
    dataset_retain = args.dataset_retain
    retain_batch = args.retain_batch
    retain_train = args.retain_train
    retain_step = args.retain_step
    retain_loss_w = args.retain_loss_w
    
    train_method = args.train_method
    norm_layer = args.norm_layer
    attack_method = args.attack_method
    component = args.component
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    save_interval = args.save_interval
    lr = args.lr
    
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    
    adv_prompt_num = args.adv_prompt_num
    attack_embd_type = args.attack_embd_type
    attack_type = args.attack_type
    attack_init = args.attack_init
    attack_step = args.attack_step
    attack_lr = args.attack_lr
    adv_prompt_update_step = args.adv_prompt_update_step
    warmup_iter = args.warmup_iter
    
    # Directory setup
    experiment_name = f'AdvUnlearn-{prompt}-method_{train_method}_{component}-Attack_{args.attack_method}-Retain_{dataset_retain}_{retain_train}_{retain_loss_w}-lr_{lr}-AttackLr_{attack_lr}-{attack_type}_adv_num_{adv_prompt_num}-{attack_embd_type}-attack_init_{attack_init}-attack_step_{attack_step}-adv_update_{adv_prompt_update_step}-warmup_iter_{warmup_iter}'
    
    run_dir = Path("./results/results_with_retaining") / args.prompt / args.dataset_retain / args.attack_method / f'AttackLr_{args.attack_lr}' / args.train_method / f'{args.component}' / args.attack_type / experiment_name / "wandb_logs"
    output_dir = Path("./results/results_with_retaining") / args.prompt / args.dataset_retain / args.attack_method /  f'AttackLr_{args.attack_lr}' / args.train_method / f'{args.component}' / args.attack_type / experiment_name

    
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    if not output_dir.exists():
        os.makedirs(str(output_dir))
        
    wandb.init(config=args,
               project=args.project_name,
               name=experiment_name,
               dir=str(run_dir),
               reinit=True)

    print_args_table(parser, args)
    
    AdvUnlearn(prompt=prompt, dataset_retain=dataset_retain, retain_batch= retain_batch, retain_train=retain_train, retain_step=retain_step, retain_loss_w=retain_loss_w, attack_method = attack_method, train_method=train_method, norm_layer=norm_layer, component=component, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, save_interval=save_interval, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, output_dir= output_dir, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, adv_prompt_num=adv_prompt_num, attack_embd_type=attack_embd_type, attack_type=attack_type, attack_init=attack_init, warmup_iter=warmup_iter, attack_step=attack_step, attack_lr=attack_lr, adv_prompt_update_step=adv_prompt_update_step)
