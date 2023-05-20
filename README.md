# NDCR
Our paper ["A Neural Divide-and-Conquer Reasoning Framework for Image Retrieval from Linguistically Complex Text"](https://arxiv.org/abs/2305.02265) have been accepted to ACL 2023.
This project aims to introduce the divide-and-conquer and neural-symbolic reasoning appraoches to handle the complex text-image reasoning problem.

## How to Run
### Environment
1. Basic Setting<br>
Python==3.7.0 (>=)
torch==1.10.1+cu111,
torchaudio==0.10.1+cu111,
torchvision==0.11.2+cu111,
transformers==4.18.0

2. unzip src_transformers.zip, volta_src.zip, CLIP.zip to the current home path. In addition, you may need download the image source from the ImageCode to the /data/game/.
We release the pre-training checkpoint about the phase 1: proposition generator and the pretraining OFA checkpoint in the Huggingface repository: https://huggingface.co/YunxinLi/pretrain_BART_generator_coldstart_OFA

3. Prepare the OFA-version Transformer<br>
git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git<br>
pip install OFA/transformers/<br>
git clone https://huggingface.co/OFA-Sys/OFA-large<br>

### Training

`python OFA_encoder_Divide_and_Conquer.py --lr 3e-5 --lr_head 4e-5 -b 32 -m ViT-B/16 -a gelu --logit_scale 1000 --add_input True --positional --frozen_clip`


## Experience
1. The training approach of large multimodal model will affect the final result on the testing set.
2. The evaluation result on the validation set may often be identical to the performance on the testing set. 
3. Adjust the random seed in the experiments may not bring some improvement, please not focus on this point.
4. This dataset is very challenging, especially for samples whose images are from the video source.
5. One significant research direction is the image modeling problem for highly similar images. It will improve the performance of NDCR.

## Acknowledge
Thanks everyone for your contributions.
If you like our work and use it in your projects, please cite our work:
```
@article{li2023neural,
  title={A Neural Divide-and-Conquer Reasoning Framework for Image Retrieval from Linguistically Complex Text},
  author={Li, Yunxin and Hu, Baotian and Ding, Yunxin and Ma, Lin and Zhang, Min},
  journal={arXiv preprint arXiv:2305.02265},
  year={2023}
}
```


