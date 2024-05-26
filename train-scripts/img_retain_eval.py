# Use the benchmark code: https://github.com/boomb0om/text2image-benchmark/tree/main
# pip install git+https://github.com/openai/CLIP.git
# pip install git+https://github.com/boomb0om/text2image-benchmark
import argparse
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from torchvision import models, transforms
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'FID_Eval',
                    description = 'Evaluate FID score')
    
    parser.add_argument('--job', help='calculate CLIP score or FID', type=str, required=False, default='fid', choices=['fid','clip'])
    parser.add_argument('--gen_imgs_path', help='generated image folder for evaluation', type=str, required=True)
    parser.add_argument('--coco_imgs_path', help='coco real image folder for evaluation', type=str, required=False, default='data/imgs/coco_10k')
    parser.add_argument('--prompt_path', help='prompt for clip score', type=str, required=False, default='data/prompts/coco_10k.csv')
    parser.add_argument('--classify_prompt_path', help='prompt for classification', type=str, required=False, default='data/prompts/imagenette_5k.csv')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    
    
    args = parser.parse_args()
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    
    if args.job == 'fid':
        from T2IBenchmark import calculate_fid
        fid, _ = calculate_fid(args.gen_imgs_path, args.coco_imgs_path)
        
        content = f'FID={fid}'
        file_path = args.gen_imgs_path+'_fid.txt'
    
    elif args.job == 'clip':
        # load CLIP
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(devices[0])
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # load CSV
        csv_file_path = args.prompt_path
        df = pd.read_csv(csv_file_path)

        # init
        clip_scores = []

        count = 0
        for index, row in df.iterrows():
            
            case_num = row['case_number']
            img_suffix = f'/{case_num}_0.png'
            img_path = args.gen_imgs_path + img_suffix
            image = Image.open(img_path)
            text = row['prompt']

            inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(devices[0])

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image

            clip_scores.append(logits_per_image.item())
            
            count += 1
            if count % 100 == 0:
                print(count)

        average_clip_score = sum(clip_scores) / (100*len(clip_scores))
        content = f'Mean CLIP Score = {average_clip_score}'
        file_path = args.gen_imgs_path+'_clip.txt'
    
    print(content)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
        if args.job == 'classify':
            for class_name in object_list:
                class_acc[class_name] = (100 * sum(class_acc[class_name])) / len(class_acc[class_name])
                file.write(f"{class_name} Acc= {class_acc[class_name]:.2f}\n")
        
# Command: python train-scripts/img_retain_eval.py --gen_imgs_path
