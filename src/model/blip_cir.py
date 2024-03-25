from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig

from src.model.blip import create_vit, init_tokenizer, load_checkpoint
from src.model.med import BertModel
from src.tools.utils import print_dist


class BLIPCir(nn.Module):
    def __init__(
        self,
        loss: Any,
        med_config="configs/med_config.json",
        image_size=384,
        vit="large",
        vit_grad_ckpt=True,
        vit_ckpt_layer=12,
        embed_dim=256,
        train_vit=False,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.loss = loss

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer
        )
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.train_vit = train_vit
        if not self.train_vit:
            # Do not train visual encoder
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        self.temp = 0.07

    def forward(self, batch, fabric):
        ref_img, tar_feat, caption, _, mmemb, description, webvid_caption = batch

        # print(mmemb.shape)
        # print(description)
        # print('='*50)

        device = ref_img.device

        if self.train_vit:
            ref_img_embs = self.visual_encoder(ref_img)
            #[bs, 577, 1024]
        else:
            with torch.no_grad():
                ref_img_embs = self.visual_encoder(ref_img) 
                #[bs, 577, 1024]

        # Encode the target image
        tar_feat = tar_feat.to(device)
        tar_img_feat = F.normalize(tar_feat, dim=-1)

        tar_mmfeat = mmemb.to(device)
        tar_img_mmfeat = F.normalize(tar_mmfeat, dim=-1)

        # Encode the reference image
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)


        ## first multi-modal feature computation goes here 

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)

        # Shift encoder
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        query_embs = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        query_feat = query_embs.last_hidden_state[:, 0, :]
        query_feat = F.normalize(self.text_proj(query_feat), dim=-1)

        ###### description loss implemented #####

        ## second multi-modal feature computation goes here 

        text2 = self.tokenizer(
            webvid_caption,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        # Shift encoder
        encoder_input_ids2 = text2.input_ids.clone()
        
        encoder_input_ids2[:, 0] = self.tokenizer.enc_token_id
        query_embs2 = self.text_encoder(
            encoder_input_ids2,
            attention_mask=text2.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        query_feat2 = query_embs2.last_hidden_state[:, 0, :]
        query_feat2 = F.normalize(self.text_proj(query_feat2), dim=-1)

        ## third multi-modal feature computation goes here 

        #
        # Add code snippet here
        #


        query_feat = query_feat*0.8 + query_feat2*0.2 # + query_feat3*0.2

        if fabric.world_size > 1:
            # d: devices, b: batch size, e: embedding dim
            query_feat = fabric.all_gather(query_feat, sync_grads=True)
            query_feat = einops.rearrange(query_feat, "d b e -> (d b) e")

            tar_img_feat = fabric.all_gather(tar_img_feat, sync_grads=True)
            tar_img_feat = einops.rearrange(tar_img_feat, "d b e -> (d b) e")

            tar_img_mmfeat = fabric.all_gather(tar_img_mmfeat, sync_grads=True)
            tar_img_mmfeat = einops.rearrange(tar_img_mmfeat, "d b e -> (d b) e")

        loss = 0.8*self.loss(query_feat, tar_img_feat, self.temp) + 0.2*self.loss(query_feat, tar_img_mmfeat, self.temp)

        return loss


def blip_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model, msg = load_checkpoint(model, ckpt_path)
        print_dist("missing keys:")
        print_dist(msg.missing_keys)
    return model
