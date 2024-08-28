conda remove -n llama --all
conda create -y -n llama python=3.10
conda activate llama

<!--
  we must use tourch 2.3.1 as torchtext is depricated and needs to be refactored
 -->
conda install pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torchtext semanticscholar spacy

python -m spacy download en_core_web_sm
python -c 'import torch; print(torch.cuda.is_available())'
python -c 'import torch; print(torch.__version__)'
