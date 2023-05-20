# NDCR
Our paper "A Neural Divide-and-Conquer Reasoning Framework for Image Retrieval from Linguistically Complex Text" have been accepted to ACL 2023.

## How to Run

### Environment
1. Basic Setting<br>
Python==3.7.0 (>=)
torch==1.10.1+cu111,
torchaudio==0.10.1+cu111,
torchvision==0.11.2+cu111,
transformers==4.18.0

unzip src_transformers.zip to current path.

2. Prepare the OFA-version Transformer<br>
git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git<br>
pip install OFA/transformers/<br>
git clone https://huggingface.co/OFA-Sys/OFA-large<br>

