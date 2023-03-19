import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np

from diffusionclip import DiffusionCLIP
from configs.paths_config import HYBRID_MODEL_PATHS
from utils.colab_utils import GoogleDrive_Dowonloader

def parse_args_and_config():
    img_path = 'imgs/celeb1.png'  
    align_face = True #param {type:"boolean"}
    edit_type = 'Pixar' #param ['Pixar', 'Neanderthal','Sketch', 'Painting by Gogh', 'Tanned',  'With makeup', 'Without makeup', 'Female → Male']
    degree_of_change = 1
    n_inv_step =  40#param{type: "integer"}
    n_test_step = 6 #param [6] 
    
    human_gdrive_ids = {'Pixar':                   ["1IoT7kZhtaoKf1uvhYhvyqzyG2MOJsqLe", "human_pixar_t601.pth"],
                    'Neanderthal':             ["1Uo0VI5kbATrQtckhEBKUPyRFNOcgwwne", "human_neanderthal_t601.pth"],
                    'Painting by Gogh':        ["1NXOL8oKTGLtpTsU_Vh5h0DmMeH7WG8rQ", "human_gogh_t601.pth"],
                    'Tanned':                  ["1k6aDDOedRxhjFsJIA0dZLi2kKNvFkSYk", "human_tanned_t201.pth"],
                    'Female → Male':           ["1n1GMVjVGxSwaQuWxoUGQ2pjV8Fhh72eh", "human_male_t401.pth"],
                    'Sketch':                  ["1V9HDO8AEQzfWFypng72WQJRZTSQ272gb", "human_sketch_t601.pth"],
                    'With makeup':             ["1OL0mKK48wvaFaWGEs3GHsCwxxg7LexOh", "human_with_makeup_t301.pth"],
                    'Without makeup':          ["157pTJBkXPoziGQdjy3SwdyeSpAjQiGRp", "human_without_makeup_t301.pth"],
                    }
    
    gid = human_gdrive_ids[edit_type][0]
    model_path = os.path.join('checkpoint', human_gdrive_ids[edit_type][1])
    dl = GoogleDrive_Dowonloader(True)
    dl.ensure_file_exists(gid, model_path)

    t_0 = int(model_path.split('_t')[-1].replace('.pth',''))
    print(f'return step t_0: {t_0}')
    exp_dir = f"runs/MANI_{img_path.split('/')[-1]}_align{align_face}"
    os.makedirs(exp_dir, exist_ok=True)
    args_dic = {
    'config': 'celeba.yml', 
    't_0': t_0, 
    'n_inv_step': int(n_inv_step), 
    'n_test_step': int(n_test_step),
    'sample_type': 'ddim', 
    'eta': 0.0,
    'bs_test': 1, 
    'model_path': model_path, 
    'img_path': img_path, 
    'deterministic_inv': 1, 
    'hybrid_noise': 0, 
    'n_iter': 1,  
    'align_face': align_face, 
    'image_folder': exp_dir,
    'model_ratio': degree_of_change,
    'edit_attr': None, 'src_txts': None, 'trg_txts': None,
    }
    
    args = dict2namespace(args_dic)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)

    # os.makedirs(args.exp, exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)
    os.makedirs('precomputed', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    # os.makedirs(args.exp, exist_ok=True)
    # args.image_folder = os.path.join(args.exp, 'image_samples')
    # if not os.path.exists(args.image_folder):
    #     os.makedirs(args.image_folder)
    # else:
    #     overwrite = False
    #     if args.ni:
    #         overwrite = True
    #     else:
    #         response = input("Image folder already exists. Overwrite? (Y/N)")
    #         if response.upper() == 'Y':
    #             overwrite = True

    #     if overwrite:
    #         # shutil.rmtree(args.image_folder)
    #         os.makedirs(args.image_folder, exist_ok=True)
    #     else:
    #         print("Output image folder exists. Program halted.")
    #         sys.exit(0)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print("<" * 80)


    runner = DiffusionCLIP(args, config)
    try:
        if args.edit_one_image:
            runner.edit_one_image()
        else:
            print('Choose one mode!')
            raise ValueError
    except Exception:
        logging.error(traceback.format_exc())


    return 0


if __name__ == '__main__':
    sys.exit(main())
