import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import argparse

import cv2
import time
import csv



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--chunk", type=str, default="0-30", help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True




conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)


model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

chunk=args.chunk

print("chunk : ", chunk)
print("="*50)


save_dir = "path_to/datasets/WebVid/8M/frames"

caption_path = "path_to/datasets/WebVid/8M/captions_13B_vicuna/{}.csv".format(chunk)
success_videos = "path_to/datasets/WebVid/8M/captions_13B_vicuna/success_{}.txt".format(chunk)


with open(caption_path,'w') as fd:
    fd.write("ID,caption")
    fd.write("\n")

with open(success_videos,'w') as fd:
    fd.write("VideoID")
    fd.write("\n")


#######################################

x,y = int(args.chunk.split('-')[0]), int(args.chunk.split('-')[1]) 


for i in range(x,y+1):

    video_path = "path_to/datasets/WebVid/8M/train/{}".format(i)
    videos = os.listdir(video_path)


    flag=1
    for vid in videos :
        t1 = time.time()
        
        vid_name = vid.split(".mp4")[0]
        try : 
            os.mkdir("{}/{}".format(save_dir, vid_name))
        except:
            pass

        cap = cv2.VideoCapture("{}/{}".format(video_path, vid))

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the middle frame index
        middle_frame_index = total_frames // 2

        # Set the frame position to the middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

        # Read the middle frame
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Could not read middle frame.")
            exit()

        # Save the middle frame as an image
        frame_path = "{}/{}/middle.jpg".format(save_dir, vid_name)
        cv2.imwrite(frame_path, frame)

        t2 = time.time()

        ########## description from minigpt4 ###########
        chat_state = CONV_VISION.copy()
        img_list = []
        chat.upload_img(frame_path, chat_state, img_list)
        user_message = "Describe the image in details."
        chat.ask(user_message, chat_state)
        description = chat.answer(  conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=1,
                max_new_tokens=300,
                max_length=2000)[0]

        description = description.replace("\n", "")

        # with open(caption_path,'a') as fd:
        #     fd.write("{},{}".format(vid_name, description))
        #     fd.write("\n")

        row = {'ID':vid_name, 'caption':description}
        with open(caption_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        with open(success_videos,'a') as fd:
            fd.write("{}".format(vid_name))
            fd.write("\n")

        t3 = time.time()
        print(flag, i,  t3-t1 )
        flag+=1