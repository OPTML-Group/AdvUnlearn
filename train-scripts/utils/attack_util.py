import torch
from .util import *
import wandb

def init_adv(k, tokenizer, all_embeddings, attack_type, device, batch = 1, attack_init_embd = None):
    # Different attack types have different initializations (Attack types: add, insert)
    adv_embedding = torch.nn.Parameter(torch.randn([batch, k, 768])).to(device)
    
    if attack_init_embd is not None:
        # Use the provided initial adversarial embedding
        adv_embedding.data = attack_init_embd[:,1:1+k].data
    else:
        # Random sample k words from the vocabulary as the initial adversarial words
        tmp_ids = torch.randint(0,len(tokenizer),(batch, k)).to(device)
        tmp_embeddings = id2embedding(tokenizer, all_embeddings, tmp_ids, device)
        tmp_embeddings = tmp_embeddings.reshape(batch, k, 768)
        adv_embedding.data = tmp_embeddings.data
    adv_embedding = adv_embedding.detach().requires_grad_(True)
    
    return adv_embedding


def soft_prompt_attack_batch(global_step, batch, words, model, model_orig, tokenizer, text_encoder, sampler, emb_0, emb_p, start_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, k, all_embeddings,  attack_round, attack_type, attack_embd_type, attack_step, attack_lr, attack_init=None, attack_init_embd = None):
    
    # print(f'======== Attack Round {attack_round} ========')
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, n_samples=batch, till_T=t, verbose=False)
    
    if attack_init == 'latest':
        adv_embedding = init_adv(k, tokenizer, all_embeddings,  attack_type, devices[0], batch, attack_init_embd)
    elif attack_init == 'random':
        adv_embedding = init_adv(k, tokenizer, all_embeddings,  attack_type, devices[0], batch)
        
    attack_opt = torch.optim.Adam([adv_embedding], lr=attack_lr)
    
    # Word Tokenization
    text_input = tokenizer(
        words, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",truncation=True
    )
    
    # Construct input_ids and input_embeds for the ESD model
    idx = 0
    id_embd_list = [] 
    for word in words:
        orig_prompt_len = len(word.split())
        
        id_embd_dict = {}
        
        sot_id, mid_id, replace_id, eot_id = split_id(text_input.input_ids[idx].unsqueeze(0).to(devices[0]), k, orig_prompt_len)
        id_embd_dict['sot_id'] = sot_id
        id_embd_dict['mid_id'] = mid_id
        id_embd_dict['replace_id'] = replace_id
        id_embd_dict['eot_id'] = eot_id
        
        # Word embedding for the prompt
        text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids[idx].unsqueeze(0).to(devices[0]), devices[0])
        sot_embd, mid_embd, _, eot_embd = split_embd(text_embeddings, k, orig_prompt_len)
        id_embd_dict['sot_embd'] = sot_embd
        id_embd_dict['mid_embd'] = mid_embd
        id_embd_dict['eot_embd'] = eot_embd
        
        id_embd_list.append(id_embd_dict)
        idx += 1
    
    if attack_embd_type == 'condition_embd':
        raise ValueError('Batch attack does not support condition_embd')
        input_adv_condition_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd)
        adv_input_ids = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)
        
    for i in range(attack_step):
        # ===== Randomly sample a time step from 0 to 1000 =====
        t_enc = torch.randint(ddim_steps, (1,), device=devices[0]) # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
        start_code = torch.randn((batch, 4, 64, 64)).to(devices[0]) # random inital noise            
    
        with torch.no_grad():
            # generate an image with the concept from ESD model
            z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, int(t_enc)) # emb_p seems to work better instead of emb_0
            # get conditional and unconditional scores from frozen model at time step t and image z
            e_0 = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]))   # [batch, 4, 64, 64]
            e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))   # [batch, 4, 64, 64]
        # breakpoint()
        
        # Construct input_ids and input_embeds for the ESD model
        if attack_embd_type == 'word_embd':
            adv_id_embd_list = []
            for j in range(batch):
                adv_id_embd_dict = {}
                input_adv_word_embedding = construct_embd(k, adv_embedding[j,:,:].unsqueeze(0), attack_type, id_embd_list[j]['sot_embd'], id_embd_list[j]['mid_embd'], id_embd_list[j]['eot_embd'])
                adv_input_id = construct_id(k, id_embd_list[j]['replace_id'], attack_type, id_embd_list[j]['sot_id'], id_embd_list[j]['eot_id'], id_embd_list[j]['mid_id'])
                adv_id_embd_dict['input_adv_word_embedding'] = input_adv_word_embedding
                adv_id_embd_dict['adv_input_id'] = adv_input_id
                adv_id_embd_list.append(adv_id_embd_dict)
            
            # combine the adversarial word embedding and input_ids for the batch
            input_adv_word_embeddings = torch.cat([adv_id_embd['input_adv_word_embedding'] for adv_id_embd in adv_id_embd_list], dim=0)
            adv_input_ids = torch.cat([adv_id_embd['adv_input_id'] for adv_id_embd in adv_id_embd_list], dim=0)
            input_adv_condition_embedding = text_encoder(input_ids = adv_input_ids.to(devices[0]), inputs_embeds=input_adv_word_embeddings)[0]
        
        # get conditional score from ESD model with adversarial condition embedding
        e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), input_adv_condition_embedding.to(devices[0]))
        e_0.requires_grad = False
        e_p.requires_grad = False
        
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        loss = criteria(e_n.to(devices[0]), e_p.to(devices[0]) )
        loss.backward()
        attack_opt.step()
        
        wandb.log({'Attack_Loss':loss.item()}, step=global_step+i)
        wandb.log({'Train_Loss': 0.0}, step=global_step+i)
    
    if attack_embd_type == 'condition_embd':
        return input_adv_condition_embedding, adv_input_ids 
    elif attack_embd_type == 'word_embd':
        return input_adv_word_embeddings, adv_input_ids 
    else:
        raise ValueError('attack_embd_type must be either condition_embd or word_embd')



  
def soft_prompt_attack(global_step, word, model, model_orig, tokenizer, text_encoder, sampler, emb_0, emb_p, start_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, k, all_embeddings,  attack_round, attack_type, attack_embd_type, attack_step, attack_lr, attack_init=None, attack_init_embd = None, attack_method='pgd'):
    
    '''
    Perform soft prompt attack on the ESD model
    Args:
        attack_type: str
            The type of attack (add or insert)
        attack_embd_type: str
            The type of adversarial embedding (condition_embd or word_embd)
        attack_step: int
            The number of steps for the attack
        attack_lr: float
            The learning rate for the attack
        attack_init: str
            The initialization method for the attack (latest or random)
        attack_init_embd: torch.Tensor
            The initial adversarial embedding
    '''
    orig_prompt_len = len(word.split())
    if attack_type == 'add':
        k = orig_prompt_len
        
    # print(f'======== Attack Round {attack_round} ========')
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)
    
    # Word Tokenization
    text_input = tokenizer(
        word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",truncation=True
    )
    sot_id, mid_id, replace_id, eot_id = split_id(text_input.input_ids.to(devices[0]), k, orig_prompt_len)
    
    # Word embedding for the prompt
    text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])
    sot_embd, mid_embd, _, eot_embd = split_embd(text_embeddings, k, orig_prompt_len)
    
    if attack_init == 'latest':
        adv_embedding = init_adv(k, tokenizer, all_embeddings,  attack_type, devices[0], 1, attack_init_embd)
    elif attack_init == 'random':
        adv_embedding = init_adv(k, tokenizer, all_embeddings,  attack_type, devices[0], 1)
    
    attack_opt = torch.optim.Adam([adv_embedding], lr=attack_lr)
    
    if attack_embd_type == 'condition_embd':
        input_adv_condition_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd)
        adv_input_ids = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)
    
    print(f'[{attack_type}] Starting {attack_method} attack on "{word}"')
    for i in range(attack_step):
        # ===== Randomly sample a time step from 0 to 1000 =====
        t_enc = torch.randint(ddim_steps, (1,), device=devices[0]) # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
        start_code = torch.randn((1, 4, 64, 64)).to(devices[0]) # random inital noise            
    
        with torch.no_grad():
            # generate an image with the concept from ESD model
            z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, int(t_enc)) # emb_p seems to work better instead of emb_0
            # get conditional and unconditional scores from frozen model at time step t and image z
            e_0 = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]))
            e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
        # breakpoint()
        
        # Construct input_ids and input_embeds for the ESD model
        if attack_embd_type == 'word_embd':
            input_adv_word_embedding = construct_embd(k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd)
            adv_input_ids = construct_id(k, replace_id, attack_type, sot_id, eot_id, mid_id)
            input_adv_condition_embedding = text_encoder(input_ids = adv_input_ids.to(devices[0]), inputs_embeds=input_adv_word_embedding)[0]
        
        # get conditional score from ESD model with adversarial condition embedding
        e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), input_adv_condition_embedding.to(devices[0]))
        e_0.requires_grad = False
        e_p.requires_grad = False
        
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        loss = criteria(e_n.to(devices[0]), e_p.to(devices[0]) )
        loss.backward()
        
        if attack_method == 'pgd':
            attack_opt.step()
        elif attack_method == 'fast_at':
            adv_embedding.grad.sign_()
            attack_opt.step()
        else:
            raise ValueError('attack_method must be either pgd or fast_at')
        
        wandb.log({'Attack_Loss':loss.item()}, step=global_step+i)
        wandb.log({'Train_Loss': 0.0}, step=global_step+i)
    
    if attack_embd_type == 'condition_embd':
        return input_adv_condition_embedding, adv_input_ids 
    elif attack_embd_type == 'word_embd':
        return input_adv_word_embedding, adv_input_ids 
    else:
        raise ValueError('attack_embd_type must be either condition_embd or word_embd')
        
def split_embd(input_embed, k, orig_prompt_len):
    sot_embd, mid_embd, replace_embd, eot_embd = torch.split(input_embed, [1, orig_prompt_len, k, 76-orig_prompt_len-k ], dim=1)
    return sot_embd, mid_embd, replace_embd, eot_embd
    
def split_id(input_ids, k, orig_prompt_len):
    sot_id, mid_id, replace_id, eot_id = torch.split(input_ids, [1, orig_prompt_len, k, 76-orig_prompt_len-k], dim=1)
    return sot_id, mid_id, replace_id, eot_id

def construct_embd(k, adv_embedding, insertion_location, sot_embd, mid_embd, eot_embd):
    if insertion_location == 'prefix_k':     # Prepend k words before the original prompt
        embedding = torch.cat([sot_embd,adv_embedding,mid_embd,eot_embd],dim=1)
    elif insertion_location == 'replace_k':  # Replace k words in the original prompt
        replace_embd = eot_embd[:,0,:].repeat(1,mid_embd.shape[1],1)
        embedding = torch.cat([sot_embd,adv_embedding,replace_embd,eot_embd],dim=1)
    elif insertion_location == 'add':      # Add perturbation to the original prompt
        replace_embd = eot_embd[:,0,:].repeat(1,k,1)
        embedding = torch.cat([sot_embd,adv_embedding+mid_embd,replace_embd,eot_embd],dim=1)
    elif insertion_location == 'suffix_k':   # Append k words after the original prompt
        embedding = torch.cat([sot_embd,mid_embd,adv_embedding,eot_embd],dim=1)
    elif insertion_location == 'mid_k':      # Insert k words in the middle of the original prompt
        embedding = [sot_embd,]
        total_num = mid_embd.size(1)
        embedding.append(mid_embd[:,:total_num//2,:])
        embedding.append(adv_embedding)
        embedding.append(mid_embd[:,total_num//2:,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding,dim=1)
    elif insertion_location == 'insert_k':   # seperate k words into the original prompt with equal intervals
        embedding = [sot_embd,]
        total_num = mid_embd.size(1)
        internals = total_num // (k+1)
        for i in range(k):
            embedding.append(mid_embd[:,internals*i:internals*(i+1),:])
            embedding.append(adv_embedding[:,i,:].unsqueeze(1))
        embedding.append(mid_embd[:,internals*(i+1):,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding,dim=1)
        
    elif insertion_location == 'per_k_words':
        embedding = [sot_embd,]
        for i in range(adv_embedding.size(1) - 1):
            embedding.append(adv_embedding[:,i,:].unsqueeze(1))
            embedding.append(mid_embd[:,3*i:3*(i+1),:])
        embedding.append(adv_embedding[:,-1,:].unsqueeze(1))
        embedding.append(mid_embd[:,3*(i+1):,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding,dim=1)
    return embedding

def construct_id(k, adv_id, insertion_location,sot_id,eot_id,mid_id):
    if insertion_location == 'prefix_k':
        input_ids = torch.cat([sot_id,adv_id,mid_id,eot_id],dim=1)
        
    elif insertion_location == 'replace_k':
        replace_id = eot_id[:,0].repeat(1,mid_id.shape[1])
        input_ids = torch.cat([sot_id,adv_id,replace_id,eot_id],dim=1)
    
    elif insertion_location == 'add':
        replace_id = eot_id[:,0].repeat(1,k)
        input_ids = torch.cat([sot_id,mid_id,replace_id,eot_id],dim=1)
    
    elif insertion_location == 'suffix_k':
        input_ids = torch.cat([sot_id,mid_id,adv_id,eot_id],dim=1)
        
    elif insertion_location == 'mid_k':
        input_ids = [sot_id,]
        total_num = mid_id.size(1)
        input_ids.append(mid_id[:,:total_num//2])
        input_ids.append(adv_id)
        input_ids.append(mid_id[:,total_num//2:])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
        
    elif insertion_location == 'insert_k':
        input_ids = [sot_id,]
        total_num = mid_id.size(1)
        internals = total_num // (k+1)
        for i in range(k):
            input_ids.append(mid_id[:,internals*i:internals*(i+1)])
            input_ids.append(adv_id[:,i].unsqueeze(1))
        input_ids.append(mid_id[:,internals*(i+1):])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
        
    elif insertion_location == 'per_k_words':
        input_ids = [sot_id,]
        for i in range(adv_id.size(1) - 1):
            input_ids.append(adv_id[:,i].unsqueeze(1))
            input_ids.append(mid_id[:,3*i:3*(i+1)])
        input_ids.append(adv_id[:,-1].unsqueeze(1))
        input_ids.append(mid_id[:,3*(i+1):])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
    return input_ids