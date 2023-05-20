# inspired from: https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP/issues/83

import json
import os
import cv2
import random
import wandb
from CLIP import clip
from CLIP.clip import model
import torch
from torch import autograd
import torch.nn.functional as F
import tqdm
from torch import nn, optim
from PIL import Image
from pathlib import Path
from collections import defaultdict
from volta_src.config import BertConfig
from volta_src.embeddings import BertLayerNorm
from volta_src.encoders import GeLU
from extras import convert_sents_to_features, BertLayer, BartAttention
import argparse
from torchvision import transforms
from src_transformers.models.bart import BartTokenizer, BartForConditionalGeneration
import sys
sys.path.append('../../')
from PIL import Image
import torch
# from torchvision import transforms
#from transformers.models.ofa import OFATokenizer
from OFA.transformers.src.transformers.models.ofa.tokenization_ofa import OFATokenizer
from OFA.transformers.src.transformers.models.ofa.modeling_ofa import OFAModel

random.seed(10)
torch.manual_seed(10)
wandb.init(project='contextualclip', settings=wandb.Settings(start_method="fork"))
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
resolution = 256
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def find_best_matches(text_features, photo_features):
    similarities = (photo_features @ text_features.T).squeeze(1)
    best_photo_idx = (-similarities).argsort()
    similarities = -similarities
    similarities.sort()
    return best_photo_idx, similarities


def convert_models_to_fp32(model):
    for p in model.parameters():
        if p.grad is not None:
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()


class MLPExpert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLPExpert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MLPTower(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPTower, self).__init__()
        # self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        # self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc2(x)
        return out


class ContextualCLIP(torch.nn.Module):
    def __init__(self, vision_backbone, bert_config, bert_model, args, width=197):
        super(ContextualCLIP, self).__init__()
        self.clip, self.preprocess = clip.load(vision_backbone, device=device, jit=False)
        self.OFA = OFAModel.from_pretrained('../../OFA-large/')
        self.OFA.to(device)
        self.text_encoder = bert_model
        self.text_encoder.to(device)
        config = BertConfig.from_dict(bert_config)
        self.fusion = args.fusion
        hidden_size = 512
        config.num_attention_heads = 8
        config.hidden_size = 768
        self.transformer_1 = nn.ModuleList([BertLayer(config) for _ in range(args.transformer_layers + 1)])
        self.transformer_1.to(device)
        self.init_rep = torch.nn.Embedding(10, config.hidden_size).to(device)
        config.hidden_size = hidden_size * 2
        self.transformer_symbolic = nn.ModuleList([BertLayer(config) for _ in range(args.transformer_layers)])
        self.transformer_symbolic.to(device)
        self.transformer_clip = nn.ModuleList([BertLayer(config) for _ in range(args.transformer_layers + 1)])
        self.transformer_clip.to(device)
        self.transformer_ = nn.ModuleList([BertLayer(config) for _ in range(args.transformer_layers)])
        self.transformer_.to(device)
        self.transformer_3 = nn.ModuleList([BertLayer(config) for _ in range(args.transformer_layers)])
        self.transformer_3.to(device)
        self.transformer_2 = nn.ModuleList([BertLayer(config) for _ in range(args.transformer_layers)])
        self.transformer_2.to(device)
        self.ver_trans = nn.ModuleList([BertLayer(config) for _ in range(args.transformer_layers)])
        self.ver_trans.to(device)
        self.batch_size = 1
        self.logit_scale = float(args.logit_scale)
        self.frozen_clip = args.frozen_clip
        self.add_input = args.add_input
        self.positional = args.positional

        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.1),
            torch.nn.Linear(1024, 1024 * 4, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.1),
            torch.nn.Linear(1024 * 4, 1024 * 1, bias=True)
            # torch.nn.ReLU(),
            # nn.Dropout(p=0.1),
            # torch.nn.Linear(512*2, 1, bias=True),
            # torch.nn.Sigmoid()
        )
        self.mapping_network_alignment.to(device)

        self.act_1 = nn.Sequential(
            nn.Dropout(p=0.1),
            torch.nn.Linear(512 * 4, 512*2, bias=True),
        )
        self.act_1.to(device)

        self.negation = nn.Sequential(
            torch.nn.Linear(512*2, 512*4, bias=True),
            torch.nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            torch.nn.Linear(512*4, 512*2, bias=False),
        )
        self.negation.to(device)
        self.bart_score = nn.Linear(768, 1, bias=True).to(device)
        self.fusion_ = nn.Sequential(
            nn.Dropout(p=0.1),
            torch.nn.Linear(512 * 4, 512 * 2, bias=True),
            nn.Dropout(p=0.1),
            torch.nn.Linear(512 * 2, 512 * 2, bias=True),
        )
        self.fusion_.to(device)
        self.selector = torch.nn.Linear(512 * 4, 512 * 2, bias=True).to(device)
        self.transfer_conjun = torch.nn.Linear(512 * 2, 512 * 4, bias=True).to(device)
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6).to(device)
        self.transfer_to_clip_space = nn.Linear(768, 1024, bias=True).to(device)
        self.transfer_to_text_space = nn.Linear(512, 768, bias=True).to(device)
        self.prediction_layer = nn.Linear(512 * 2, 1).to(device)
        self.prediction_layer_1 = nn.Linear(512 * 2, 1, bias=True).to(device)
        self.prediction_layer_3 = nn.Linear(512 * 2, 1, bias=True).to(device)
        self.prediction_layer_4 = nn.Linear(512 * 2, 1, bias=True).to(device)
        self.prediction_layer_symbolic = nn.Linear(512 * 2, 1, bias=True).to(device)
        self.init_rep_clip = torch.nn.Embedding(10, 512).to(device)
        self.modifier = nn.Sequential(
            torch.nn.Linear(512 * 4, 512 * 4, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            torch.nn.Linear(512 * 4, 512 * 2, bias=True),
        )
        self.modifier.to(device)
        self.cross_copy = BartAttention(config.hidden_size * 2, 1, dropout=0.01, is_decoder=True)
        self.cross_copy.to(device)
        self.act_sig = torch.nn.Linear(config.hidden_size * 4, 1, bias=True).to(device)
        self.pred_weight = torch.nn.Linear(512 * 2, 1, bias=True).to(device)
        if args.positional:
            self.positional_emb = torch.nn.Embedding(10, hidden_size * 2).to(device)
            self.condition_symbolic = torch.nn.Embedding(10, hidden_size * 2).to(device)
        config.hidden_size = 512 * 4
        config.num_attention_heads = 8
        self.tau = nn.Linear(1, 1, bias=True).to(device)
        self.gnn = nn.ModuleList([BertLayer(config) for _ in range(1)])
        self.gnn.to(device)

    def forward(self, images, text, pos_mask=None, sentence_split=None, img_type=None, text_=None, input_ids=None):
        global_image = images
        # image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        if self.frozen_clip:
            with torch.no_grad():
                input_ids_context = input_ids[:1].repeat(10, 1)
                gen = self.OFA(input_ids_context, patch_images=global_image, decoder_input_ids=None)
                all_hidden_states = gen.encoder_last_hidden_state
                all_hidden_states = all_hidden_states[:, 256]
                # input_ids_conditions = input_ids[1:]
                # gen = self.OFA(input_ids_conditions, patch_images=None, decoder_input_ids=None)
                # condition_hidden_states = gen.encoder_last_hidden_state
                # condition_hidden_states = condition_hidden_states[:, 0]
                input_ids_cls = input_ids_context[:, :1]
                image_gen = self.OFA(input_ids_cls, patch_images=global_image, decoder_input_ids=None)
                image_hidden_states = image_gen.encoder_last_hidden_state
                # image_hidden_states_seq = image_hidden_states[:, :256]
                image_hidden_states = image_hidden_states[:, 256]
        else:
            input_ids = input_ids[:1].repeat(10, 1)
            gen = self.OFA(input_ids, patch_images=global_image, decoder_input_ids=None)
            all_hidden_states = gen.encoder_last_hidden_state
            all_hidden_states = all_hidden_states[:, 256]
            input_ids = input_ids[:, :1]
            image_gen = self.OFA(input_ids, patch_images=global_image, decoder_input_ids=None)
            image_hidden_states = image_gen.encoder_last_hidden_state
            image_hidden_states = image_hidden_states[:, 256]

        #### proposition generator
        with torch.no_grad():
            encoder_outputs = self.text_encoder.model.encoder(
                input_ids=text['input_ids'],
                attention_mask=text['attention_mask'],
            )
            t_fine = encoder_outputs[0][:, 1:, :][:, :-1, :]
            #text_features_ = encoder_outputs[0][:, 0, :]
            rep_ = self.init_rep(torch.arange(10).to(device))
            rep_ = rep_.unsqueeze(0)
            attention_mask = torch.tril(torch.ones((1, 1, 10, 10))).to(device)
            attention_mask.masked_fill_(attention_mask == 0.0, -1e4)
            attention_mask.masked_fill_(attention_mask == 1.0, 0.0)
            # attention_mask = None
            for lm in self.transformer_1[0:]:
                rep_ = lm(hidden_states=rep_, attention_mask=attention_mask, encoder_out=t_fine[:1].half())
            text_all = rep_.squeeze(0)
            cos_len = min(text['input_ids'].size(0) - 1, 4)
        # training the mapping linear layer to the clip text space
        text_features_bart_bart = self.transfer_to_clip_space(text_all[:cos_len].half())
        fine_rep = text_features_bart_bart
        loss_cos = 0.0
        text_features_norm_2 = fine_rep / fine_rep.norm(dim=-1, keepdim=True)
        if cos_len > 1:
            loss_cos = loss_cos + max(torch.mean(self.cos_sim(text_features_norm_2[:-1], text_features_norm_2[1:]) - 0.2), 0.0)

        #### context-to-image driven signal
        with torch.no_grad():
            x_ = all_hidden_states.unsqueeze(0).half()
            # pred_context_con = self.prediction_layer_1(x_).squeeze(-1)
            if self.positional:
                embs = self.positional_emb(torch.arange(10).to(device))
                embs = embs * pos_mask
                x_pos = x_ + embs
            else:
                x_pos = x_
            attention_mask = torch.ones((self.batch_size, 1, 1, 10)).to(device)
            x_c = self.transformer_3[0](hidden_states=x_pos, attention_mask=attention_mask)
            for layer_module in self.transformer_3[1:]:
                x_c = layer_module(hidden_states=x_c, attention_mask=attention_mask)
            x_context = x_c.half() + x_
            pred_context = self.prediction_layer_3(x_context).squeeze(-1)

        # System1: conditions-to-image
        # Fusion 1
        text_norm_final = torch.cat(10 * [fine_rep.unsqueeze(1)], dim=1)
        images_norm_final = torch.stack(text_norm_final.size(0) * [image_hidden_states])
        p_fus = self.fusion_(torch.cat([text_norm_final, images_norm_final], dim=-1).half())
        # Fusion 2
        # text_norm_final = torch.cat(10 * [fine_rep.unsqueeze(1)], dim=1)
        # text_norm_final = self.mapping_network_alignment(text_norm_final)
        # fusion_list = []
        # for _ in range(text_norm_final.size(0)):
        #     prefix_embeddings = text_norm_final[_].reshape(text_norm_final[_].size(0), 1, 1024)
        #     cur_ids = input_ids[1:][_:_+1].repeat(10, 1)[:, :20]
            #tmp_out = self.OFA(input_ids_context[:, :1].repeat(1, prefix_embeddings.size(1) + 1), patch_images=global_image, prefix_embeddings=prefix_embeddings, decoder_input_ids=None)
            # tmp_out = self.OFA(cur_ids, patch_images=global_image, prefix_embeddings=prefix_embeddings, decoder_input_ids=None)
            # tmp_out = self.transformer_[0](hidden_states=text_norm_final[_].unsqueeze(1), inferer_out= image_hidden_states_seq)
            # tmp_out = self.transformer_[1](hidden_states=tmp_out, inferer_out=image_hidden_states_seq)
            #text_norm_final[_].unsqueeze(1)
            # fusion_list.append(tmp_out.encoder_last_hidden_state[:, 256])
        # p_fus = torch.stack(fusion_list)

        x_ = p_fus
        if self.positional:
            embs = self.condition_symbolic(torch.arange(10).to(device))
            embs = embs * pos_mask
            x_pos = x_ + embs
        else:
            x_pos = x_
        # attention_mask = torch.ones((1, 1, 1, 10)).to(device)
        x_c = self.transformer_2[0](hidden_states=x_pos, attention_mask=None)
        for layer_module in self.transformer_2[1:]:
            x_c = layer_module(hidden_states=x_c, attention_mask=None)
        x_final = x_c.half()

        ##########prediction of conditions
        # pred_condition_init = self.prediction_layer_4(x_final).squeeze(-1)
        # if pred_condition_init.size(0) < 5:
        #     pred_embed = torch.cat([pred_condition_init] + (5 - pred_condition_init.size(0)) * [self.final_embedding.unsqueeze(0)], dim=0)
        # else:
        #     pred_embed = pred_condition_init[:5]
        # pred_embed = pred_embed.view(1, -1)
        # pred_condition_init_total = self.pred_init(pred_embed.half())
        # pred_condition_init_total = torch.sum(2 * F.softmax(pred_condition_init / 0.5, dim=-1), dim=0, keepdim=True)
        # pred_condition_init_total_pres = self.prediction_layer_4(x_.half()).squeeze(-1)
        # pred_condition_init_total_pre = torch.sum(2 * F.softmax(pred_condition_init_total_pres / 0.5, dim=-1), dim=0, keepdim=True)

        ######################################################

        pred_f = []
        x_context_ = x_context.detach()
        ####### System 2: Neural-Symbolic Reasoner
        x_context_l = torch.cat(x_final.size(0) * [x_context_.half()], dim=0)
        x_final_conditions = self.modifier(torch.cat([x_final, x_context_l], dim=-1))
        pred_condition_t = self.prediction_layer_4(x_final_conditions.half()).squeeze(-1)

        # Negation executor
        nega_r = self.negation(x_final_conditions.half().detach())
        nega_pred = self.prediction_layer_4(nega_r.half()).squeeze(-1)
        nega_pred_norm = F.softmax(nega_pred.detach() / 0.2, dim=-1)
        nega_pred_images = torch.mm(nega_pred_norm, image_hidden_states.half()).unsqueeze(1).repeat(1, 10, 1)
        nega_pred_latent_cur = torch.cat([nega_r, nega_pred_images], dim=-1)

        # Conjunction
        pred_result_norm = F.softmax(pred_condition_t.detach() / 0.2, dim=-1)
        pred_condition_images = torch.mm(pred_result_norm, image_hidden_states.half()).unsqueeze(1).repeat(1, 10, 1)
        x_context_rep = self.transfer_conjun(x_context_.half())
        x_final_cur = torch.cat([x_final_conditions, pred_condition_images], dim=-1)
        # with negation module
        x_final_cur = torch.cat([x_final_cur, nega_pred_latent_cur], dim=0)
        # without negation module
        #x_final_cur = x_final_cur
        act_rep = torch.cat([x_context_rep, x_final_cur], dim=0)
        act_rep = self.gnn[0](hidden_states=act_rep, image_out=x_final_cur.transpose(0, 1))
        x_context_final = self.selector(act_rep[:1].half())
        ########################################################################

        ####### Combine System 1 and System 2
        x_copy_l = x_final_conditions.half().detach()
        x_copy_l = torch.bmm(F.softmax(torch.sigmoid(self.pred_weight(x_copy_l).transpose(1, 2)), dim=-1), x_copy_l).transpose(0, 1)
        condition_img = torch.mm(F.softmax(pred_condition_t.detach() / 0.20, dim=-1), image_hidden_states.half())
        x_copy_l = torch.cat([x_copy_l, condition_img.unsqueeze(0)], dim=-1)
        x_context_q = x_context_final.half().detach()
        x_context_q = torch.bmm(F.softmax(torch.sigmoid(self.pred_weight(x_context_q).transpose(1, 2)), dim=-1), x_context_q).squeeze(1)
        pred_context_con = self.prediction_layer(x_context_final.half()).squeeze(-1)
        context_s_images = torch.mm(F.softmax(pred_context_con.detach() / 0.20, dim=-1), image_hidden_states.half())
        x_context_q = torch.cat([x_context_q, context_s_images], dim=-1)
        pred_result_norm_copy_l = F.softmax(pred_condition_t.detach() / 0.5, dim=-1)
        score_matrix = self.cross_copy(x_context_q.unsqueeze(0), x_copy_l, output_attentions=True, tau=0.5, input_type=False)
        conditions_pre = torch.mm(score_matrix[1].squeeze(0).squeeze(0), pred_result_norm_copy_l)
        sig_act = torch.sigmoid(self.act_sig(torch.cat([score_matrix[0].squeeze(0), x_context_q], dim=-1)))
        pred_final = sig_act * conditions_pre + (1 - sig_act) * F.softmax(pred_context_con.detach() / 0.5, dim=-1)

        ####### output logits
        pred_f.append(pred_condition_t)
        pred_f.append(pred_context_con)
        pred_f.append(pred_final)
        pred_f_agg = torch.cat(pred_f, dim=0)

        ########## return final results
        return pred_final, torch.cat([pred_context, pred_condition_t], dim=0), nega_pred, pred_f_agg, loss_cos

        ##### Return System 1
        # pred_condition_t_final = torch.sum(2 * F.softmax(pred_condition_t, dim=-1), dim=0, keepdim=True)
        # return pred_condition_t_final, torch.cat([pred_condition_t_final, pred_condition_t], dim=0), \
        #      None, None, loss_cos

        ######## return training phrase 1
        # return pred_condition_init_total, torch.cat([pred_condition_init_total_pre, pred_condition_init], dim=0), \
        #       None, None, loss_cos

        ######## return strong baseline
        # return pred_context_con, None, None, None, None
config = wandb.config
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batchsize", type=int)
parser.add_argument("--lr_head", type=float)
parser.add_argument("--lr", type=float)
parser.add_argument("-m", "--model", type=str, default='ViT-B/16')
parser.add_argument("--fusion", type=str, default='mult')
parser.add_argument("-a", "--activation", default='relu')
parser.add_argument("-s", "--logit_scale", default=1)
parser.add_argument("--frozen_clip", action="store_true")
parser.add_argument("--finetuned_checkpoint_path",
                    default='./checkpoints/System_1_and_2_neural_calculator_end2end_OFA_training_phrase1_1_2_0.0582.pt')
# ./checkpoints/CONTRA_clip_best_0.2967.pt
parser.add_argument("--add_input", type=bool, default=True)
parser.add_argument("--positional", action="store_true")
parser.add_argument("--head_scheduler", default=0.95, type=float)
parser.add_argument("--base_scheduler", default=0.95, type=float)
parser.add_argument("--transformer_layers", default=2, type=int)
parser.add_argument("--all_pos", action="store_true")
parser.add_argument("--test", type=bool, default=False)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--valid_descr_path', type=str, default='../../data/valid_data.json')
parser.add_argument('--test_descr_path', type=str, default='../../data/test_data_unlabeled.json')
parser.add_argument('--train_descr_path', type=str, default='../../data/train_data.json')
parser.add_argument('--imgs_path', type=str, default='../../data/games')
parser.add_argument("--job_id")

args = parser.parse_args()
assert args.fusion in ['concat', 'mult']
assert args.activation in ['leaky-relu', 'relu', 'gelu']
wandb.config.update(args)
img_dirs = args.imgs_path
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f'DEVICE USED: {device}')
OFA_tokenizer = OFATokenizer.from_pretrained('../../OFA-large/')
tokenizer = BartTokenizer.from_pretrained('./bart_base_checkpoints')
bart_model = BartForConditionalGeneration.from_pretrained('./bart_base_checkpoints')
bert_config = json.load(open('vilbert-and-bert-config.json', 'r'))
contextual_clip = ContextualCLIP(args.model, bert_config, bart_model, args)
if args.finetuned_checkpoint_path:
    checkpoint = torch.load(args.finetuned_checkpoint_path, map_location='cpu')
    if not args.test:
        contextual_clip.load_state_dict(checkpoint['model_state_dict'], False)
        # copy_checkpoints = checkpoint['model_state_dict'].copy()
        # for n, p in copy_checkpoints.items():
        #     if 'OFA.' not in n and 'transformer_3.' not in n and 'positional_emb.' not in n and 'prediction_layer_3.' not in n:
        #         del checkpoint['model_state_dict'][n]
        # contextual_clip.load_state_dict(checkpoint['model_state_dict'], False)
        # checkpoint = torch.load('./checkpoints/System_1_and_2_neural_calculator_end2end_OFA_final_2_0.2285.pt', map_location='cpu')
        # copy_checkpoints = checkpoint['model_state_dict'].copy()
        # for n, p in copy_checkpoints.items():
        #     if 'transformer_3.' not in n and 'positional_emb.' not in n and 'prediction_layer_3.' not in n:
        #         del checkpoint['model_state_dict'][n]
        # contextual_clip.load_state_dict(checkpoint['model_state_dict'], False)
        # clip_checkpoints = torch.load('./checkpoints/CONTRA_clip_P16_best_2_0.2958.pt')
        # contextual_clip.clip.load_state_dict(clip_checkpoints['model_state_dict'])
        # checkpoint = torch.load('./checkpoints/CONTEXTUAL_logic_2_best_logic_context_premises_decomposition_pretraining_phase_fourth_5_.pt', map_location='cpu')
        # copy_checkpoints = checkpoint['model_state_dict'].copy()
        # for n, p in copy_checkpoints.items():
        #     if 'text_encoder.' not in n and 'init_rep.' not in n and 'transformer_1.' not in n:
        #         del checkpoint['model_state_dict'][n]
        # contextual_clip.load_state_dict(checkpoint['model_state_dict'], False)
    else:
        contextual_clip.load_state_dict(checkpoint['model_state_dict'])
# for n, p in contextual_clip.clip.named_parameters():
#     if 'resblocks.11.' in n or n in ['text_projection', 'visual.proj']:
#         p.requires_grad = True
#     else:
#         p.requires_grad = False
for n, p in contextual_clip.OFA.named_parameters():
    p.requires_grad = False

config = wandb.config
wandb.watch(contextual_clip)
if device == "cpu":
    contextual_clip.float()
else:
    clip.model.convert_weights(
        contextual_clip)  # Actually this line is unnecessary since clip by default already on float16
    contextual_clip.text_encoder.float()
    contextual_clip.OFA.float()
MAX_EPOCHS = 40
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
loss_nn = nn.NLLLoss()
loss_func = nn.MSELoss()
head_params = list(contextual_clip.prediction_layer_1.parameters()) + list(contextual_clip.prediction_layer.parameters()) \
              + list(contextual_clip.transformer_.parameters()) + list(contextual_clip.prediction_layer_4.parameters()) \
              + list(contextual_clip.prediction_layer_3.parameters()) + list(contextual_clip.transformer_symbolic.parameters()) \
              + list(contextual_clip.act_1.parameters()) + list(contextual_clip.gnn.parameters()) \
              + list(contextual_clip.transformer_3.parameters()) + list(contextual_clip.transformer_2.parameters()) \
              + list(contextual_clip.negation.parameters())\
              + list(contextual_clip.transformer_clip.parameters()) + list(contextual_clip.transfer_conjun.parameters()) \
              + list(contextual_clip.prediction_layer_symbolic.parameters()) + list(contextual_clip.fusion_.parameters()) \
              + list(contextual_clip.selector.parameters()) + list(contextual_clip.modifier.parameters()) \
              + list(contextual_clip.bart_score.parameters()) + list(contextual_clip.transfer_to_clip_space.parameters()) \
              + list(contextual_clip.cross_copy.parameters()) + list(contextual_clip.act_sig.parameters()) \
              + list(contextual_clip.pred_weight.parameters()) + list(contextual_clip.mapping_network_alignment.parameters())
if args.positional:
    head_params += list(contextual_clip.positional_emb.parameters())
    head_params += list(contextual_clip.condition_symbolic.parameters())

#pretrained_params = list(contextual_clip.OFA.parameters())
pretrained_params = list(contextual_clip.text_encoder.parameters()) + \
                    list(contextual_clip.transformer_1.parameters()) + list(contextual_clip.init_rep.parameters())

optimizer = optim.Adam([{"params": pretrained_params}, {"params": head_params, "lr": config.lr_head}], lr=config.lr,
                       betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)
# optimizer = optim.SGD(params=contextual_clip.parameters(), lr=config.lr)
lambda1 = lambda epoch: args.base_scheduler ** epoch
lambda2 = lambda epoch: args.head_scheduler ** epoch
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
best_val = 0
args.test = False

# test_data = json.load(open(args.test_descr_path, 'r'))
# test_data_div = json.load(open('../../data/test_model_split_2.json', 'r'))
# test_data_label = json.load(open('./results/human_labeling_test_set.json', 'r'))
# test = []
# for img_dir, data in test_data_label.items():
#     for img_idx, text in data.items():
#         test.append((img_dir, img_idx, text))
# print('Total number of Test: ', len(test))

if not args.test:
    valid_data = json.load(open(args.valid_descr_path, 'r'))
    valid_data_div = json.load(open('../../data/dev_model_split_2.json', 'r'))
    train_data = json.load(open(args.train_descr_path, 'r'))
    train_data_div = json.load(open('../../data/train_model_split_2.json', 'r'))
    train = []
    train_group = []
    for img_dir, data in train_data.items():
        current_group = []
        for img_idx, text in data.items():
            train.append((img_dir, img_idx, text))
            current_group.append((img_dir, img_idx, text))
        train_group.append(current_group)
    print('Total number of Train: ', len(train))
    train_videos = []
    for img_dir, data in train_data.items():
        if 'open-images' in str(img_dir):
            continue
        for img_idx, text in data.items():
            train_videos.append((img_dir, img_idx, text))
    print('Video of Train: ', len(train_videos))
    train_images = []
    for img_dir, data in train_data.items():
        if 'open-images' not in str(img_dir):
            continue
        for img_idx, text in data.items():
            train_images.append((img_dir, img_idx, text))
    print('Static images of Train: ', len(train_images))
    valid = []
    for img_dir, data in valid_data.items():
        for img_idx, text in data.items():
            valid.append((img_dir, img_idx, text))
    print('Total number of Valid: ', len(valid))
    train = train
    print(len(train))
    for i in range(args.epochs):
        save_model = False
        # EVALUATE
        if i >= 1:
            contextual_clip.eval()
            correct = 0
            correct_c = 0
            correct_s = 0
            correct_both = 0
            correct_all_false = 0
            context_in_conditions_all = 0
            context_in_conditions = 0
            context_in_conditions_add = 0
            distance_pred_pred_s = 0
            distance_pred_pred_c = 0
            distance_pred_ = 0
            correct_c_only = 0
            correct_max = 0
            correct_impro = 0
            length_improv = 0
            correct_distance = 0
            length_distance = 0
            cond_length = dict()
            img_total = 0
            vid_total = 0
            img_correct = 0
            vid_correct = 0
            ranks = defaultdict(int)
            eval_step = 0
            for img_dir, img_idx, text in tqdm.tqdm(valid):
                text = [text]
                img_idx = int(img_idx)
                img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
                img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
                images = [Image.open(photo_file) for photo_file in img_files]
                # image_l = [cv2.imread('/'.join(photo_file.parts)) for photo_file in img_files]
                images = torch.stack([patch_resize_transform(photo) for photo in images]).to(device)
                text_length = []
                text_cond = []
                text_text = []
                # text_split = valid_data_div[text[0]]
                if text[0] in valid_data_div.keys():
                    text_split = valid_data_div[text[0]]
                else:
                    print('Errors!!!')
                    break
                for s in text_split:
                    all_l = (text[0] + s).split()
                    text_cond.append(s)
                    text_text.append(text[0] + ' <s> ' + s)
                text = text + text_cond
                if len(text_cond) not in cond_length.keys():
                    cond_length[len(text_cond)] = 1
                else:
                    cond_length[len(text_cond)] += 1
                eval_step += 1
                #text_ = clip.tokenize(text, truncate=True).to(device)
                input_ids = OFA_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=180)['input_ids'].to(device)
                text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=180).to(device)
                if "open-images" in str(img_dir):
                    pos_mask = torch.zeros((10, 1)).to(device)
                else:
                    pos_mask = torch.ones((10, 1)).to(device)
                with torch.no_grad():
                    logits, text_f, true_score, img_text, neural_pre = contextual_clip(images, text, pos_mask, None,
                                                                                       str(img_dir), text_=None, input_ids=input_ids)
                pred_ = torch.argmax(logits, dim=-1)
                pred = [pred_.item()]
                pred_c = text_f[:1]
                context_sort, sort_index = torch.sort(pred_c.squeeze(0))
                pred_c_top_2 = sort_index.cpu().numpy().tolist()[-2:]
                pred_c = [torch.argmax(pred_c, dim=-1).item()]
                pred_all = pred + pred_c
                pred_s = text_f[1:]
                #print(text_f)
                pred_s = torch.argmax(pred_s, dim=-1)
                pred_s = [pred_s[_].item() for _ in range(pred_s.size(0))][:2]
                if len(list(set(pred_s))) == 1 and pred_c[0] in pred_s and pred[0] not in pred_s:
                    if img_idx in pred_c:
                        distance_pred_pred_c += abs(pred_s[0] - pred[0])
                        distance_pred_ += 1
                    if img_idx in pred:
                        distance_pred_ += 1
                        distance_pred_pred_s += abs(pred_s[0] - pred[0])
                    #pred = pred_c
                if img_idx in pred_c:
                    correct_c += 1
                    if pred_c[0] in pred_s:
                        context_in_conditions_all += 1
                if img_idx in pred_all:
                    correct_max += 1
                if pred_c[0] not in pred_s:
                    if img_idx in pred_c:
                        correct_c_only += 1
                if img_idx not in pred_c and img_idx in pred:
                    if pred_c[0] not in pred_s:
                        context_in_conditions += 1
                    correct_impro += 1
                    length_improv += len(text_cond)
                if img_idx not in pred and img_idx in pred_c:

                    correct_distance += 1
                    length_distance += len(text_cond)
                if img_idx in pred and img_idx in pred_c:
                    correct_both += 1
                if img_idx not in pred and img_idx not in pred_c:
                    correct_all_false += 1
                if img_idx in pred_s:
                    correct_s += 1
                if img_idx in pred:
                    correct += 1
                if 'open-images' in img_dir:
                    img_total += 1
                    if img_idx in pred:
                        img_correct += 1
                else:
                    vid_total += 1
                    if img_idx in pred:
                        vid_correct += 1
            acc = correct / eval_step
            image_acc = img_correct / img_total
            video_acc = vid_correct / vid_total
            context_acc = correct_c / eval_step
            print('Total ACC: ', acc)
            print('context ACC: ', correct_c / eval_step)
            print('conditions ACC: ', correct_s / eval_step)
            print('Max ACC: ', correct_max / eval_step)
            print('Improve ACC on the context: ', correct_impro, length_improv)
            print('Improve ACC on the context with context not in conditions: ', context_in_conditions,
                  context_in_conditions_add)
            print('Distance between prediction and pred_context: ', distance_pred_, distance_pred_pred_c,
                  distance_pred_pred_s)
            print('context right with context in conditions: ', context_in_conditions_all, correct_c)
            print('Distance between context and conditions: ', correct_distance, length_distance)
            print('Both correct: ', correct_both, correct_both / eval_step)
            print('Both False: ', correct_all_false, correct_all_false / eval_step)
            print('Image ACC: ', img_correct / img_total)
            # print('Image total: ', img_total)
            print('Video Acc: ', vid_correct / vid_total)
            print(cond_length)
            wandb.log({'val_acc': acc})
            # video_acc max acc!!!!
            if acc > best_val:
                best_val = acc
                save_model = True
                string = ''
                for key, val in list(vars(args).items()):
                    if 'path' not in key:
                        string += f'_{val}'
                torch.save({
                    'epoch': i,
                    'model_state_dict': contextual_clip.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                    f"checkpoints/System_1_and_2_neural_calculator_end2end_OFA_{i}_{round(best_val, 4)}.pt")
            print('------------------------------')
        print(f'EPOCH: {i}')
        step = 0
        global_loss = 0.0
        train_c = train.copy()
        random.shuffle(train)
        while train_c == train:
            random.shuffle(train)
        contextual_clip.train()
        batch_number = 0
        images_train = 0
        for img_dir, img_idx, text in train:
            step += 1
            text = [text]
            img_idx = int(img_idx)
            img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            images = [Image.open(photo_file) for photo_file in img_files]
            images = torch.stack([patch_resize_transform(photo) for photo in images]).to(device)
            text_word = text[0]
            text_length = []
            text_cond = []
            text_text = []
            text_split = []
            if text[0] in train_data_div.keys():
                text_split = train_data_div[text[0]]
            for s in text_split:
                all_l = (text[0] + s).split()
                text_cond.append(s)
                text_text.append(text[0] + ' <s> ' + s)

            text = text + text_cond
            # text = text
            #text_ = clip.tokenize(text, truncate=True).to(device)
            input_ids = OFA_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=180)['input_ids'].to(device)
            text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=180).to(device)
            if "open-images" in str(img_dir):
                pos_mask = torch.zeros((10, 1)).to(device)
            else:
                pos_mask = torch.ones((10, 1)).to(device)
            if args.all_pos:
                pos_mask = torch.ones((10, 1)).to(device)
            logits, p_condition, p_condition_single, img_text, loss_contra = \
                contextual_clip(images, text, pos_mask, None, str(img_dir), text_=None, input_ids=input_ids)
            ground_truth = torch.tensor([img_idx]).long().to(device)
            loss_init = loss_txt(logits, ground_truth)

            logits_softmax = F.softmax(p_condition_single, dim=-1)
            l = torch.nn.KLDivLoss(reduction='batchmean')
            loss_pc_neg = max(0.2 - l(logits_softmax.log(), F.softmax(img_text[:-2].detach() / 0.25, dim=-1)), 0.0)
            ground_truth = torch.tensor(img_text.size(0) * [img_idx]).long().to(device)
            loss_pc_3 = loss_txt(img_text[:-1], ground_truth[:-1])
            loss = loss_init + loss_pc_3 + loss_pc_neg

            if loss != 0.0 and torch.isnan(loss):
                print(loss)
                break
            if loss > 0.0:
                if 'open-images' in str(img_dir):
                    images_train += 1
                batch_number += 1
                loss.backward()
                global_loss += loss.item()
            if batch_number > 0 and batch_number % config.batchsize == 0:
                print(f'TOTAL LOSS: {global_loss / step}')
                print('STEP: ' + str(step))
                wandb.log({'loss': loss})
                if device == "cpu":
                    optimizer.step()
                else:
                    convert_models_to_fp32(contextual_clip)
                    optimizer.step()
                    clip.model.convert_weights(contextual_clip)
                    contextual_clip.text_encoder.float()
                    contextual_clip.OFA.float()
                optimizer.zero_grad()
                contextual_clip.zero_grad()
                batch_number = 0
            contextual_clip.train()
        print('Training Images: ', images_train)
        scheduler.step()

else:
    test_data = json.load(open(args.test_descr_path, 'r'))
    test_data_div = json.load(open('../../data/test_model_split_2.json', 'r'))
    test_data_label = json.load(open('./results/human_labeling_test_set.json', 'r'))
    test = []
    for img_dir, data in test_data_label.items():
        for img_idx, text in data.items():
            test.append((img_dir, img_idx, text))
    print('Total number of Test: ', len(test))
    results = defaultdict(dict)
    conditions_number = defaultdict(dict)
    contextual_clip.eval()
    correct = 0
    img_total = 0
    vid_total = 0
    img_correct = 0
    vid_correct = 0
    test_step = 0
    ranks = defaultdict(int)
    for img_dir, img_idx, text in tqdm.tqdm(test):
        curr_t = text
        text = [text]
        # if test_step == 200:
        #     break
        # print(text)
        img_idx = int(img_idx)
        img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
        img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
        images = [Image.open(photo_file) for photo_file in img_files]
        # image_l = [cv2.imread('/'.join(photo_file.parts)) for photo_file in img_files]
        images = torch.stack([patch_resize_transform(photo) for photo in images]).to(device)
        text_length = []
        text_cond = []
        text_text = []
        text_split = test_data_div[text[0]]
        negation_s = []
        for s in text_split:
            all_l = (text[0] + s).split()
            text_cond.append(s)
            text_text.append(text[0] + ' <s> ' + s)
            # print(s)
        # print('-----------------------------------')
        # print(results)
        test_step += 1
        if len(text_cond) not in conditions_number.keys():
            conditions_number[len(text_cond)] = 1
        else:
            conditions_number[len(text_cond)] += 1
        text = text + text_cond
        input_ids = OFA_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=180)[
            'input_ids'].to(device)
        text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=180).to(device)
        if "open-images" in str(img_dir):
            pos_mask = torch.zeros((10, 1)).to(device)
        else:
            pos_mask = torch.ones((10, 1)).to(device)
        with torch.no_grad():
            logits, p_condition, p_condition_single, img_text, pre_context = \
                contextual_clip(images, text, pos_mask, None, str(img_dir), text_=None, input_ids=input_ids)

        pred_ = torch.argmax(logits, dim=-1)
        pred = [pred_.item()]
        if img_idx in pred:
            correct += 1
        if 'open-images' in img_dir:
            img_total += 1
            if img_idx in pred:
                img_correct += 1
        else:
            vid_total += 1
            if img_idx in pred:
                vid_correct += 1
        if img_dir not in results.keys():
            results[img_dir] = []
        results[img_dir].append(int(pred[0]))
        # results[img_dir][curr_t] = int(pred[0])
        # print(results)
    acc = correct / test_step
    print('Total ACC: ', acc)
    print('Image ACC: ', img_correct / img_total)
    print('Video Acc: ', vid_correct / vid_total)
    print('Image Number: ', img_total)
    print('Video Number: ', vid_total)
    print('Conditions Number: ', conditions_number)
    #json.dump(results, open(f'./results/NDCR_test.json', 'w'), indent=2)
