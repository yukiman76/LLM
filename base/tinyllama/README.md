git clone https://github.com/jzhang38/TinyLlama.git tinyllama

need to get the datasets

Get https://huggingface.co/datasets/bigcode/starcoderdata

git clone https://huggingface.co/datasets/bigcode/starcoderdata starcoderdata

Get https://huggingface.co/datasets/cerebras/SlimPajama-627B

git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B slimpajama-627b


Create Conda env

conda remove -n tinyllama --all
conda create -y -n tinyllama python=3.10
conda activate tinyllama
conda install pytorch==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

Lets test to make sure we have GPU support
python -c 'import torch; print(torch.cuda.is_available())'
python -c 'import torch; print(torch.__version__)'

now we can install all the core requremetns

pip install ninja -U
pip install flash-attn --no-build-isolation
pip install -r requirements.txt tokenizers sentencepiece


now we have to setup the data


python scripts/prepare_starcoder.py --source_path /data/tinyllama/starcoderdata/ --tokenizer_path data/llama --destination_path /data/tinyllama/slim_star_combined--split train --percentage 1.0

python scripts/prepare_slimpajama.py --source_path /data/tinyllama/SlimPajama --tokenizer_path data/llama  --destination_path /data/tinyllama/slim_star_combined --split validation --percentage 1.0

python scripts/prepare_slimpajama.py --source_path /data/tinyllama/SlimPajama --tokenizer_path data/llama  --destination_path /data/tinyllama/slim_star_combined --split train --percentage 1.0


lightning run model \
    --node-rank=0  \
    --main-address=172.16.101.5 \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=2 \
    pretrain/tinyllama.py --devices 8 --train_data_dir data/slim_star  --val_data_dir data/slim_star
