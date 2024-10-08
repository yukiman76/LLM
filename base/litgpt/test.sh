echo "Getting Data"
git clone --depth=1 --branch=main https://github.com/mlschmitt/classic-books-markdown data && rm -rf data/.git
# 1) Download the tokenizer
echo "Download the tokenizer"
litgpt download EleutherAI/pythia-160m \
  --tokenizer_only True

# 2) Pretrain the model
echo "Pretrain the model"
litgpt pretrain pythia-160m \
  --tokenizer_dir EleutherAI/pythia-160m \
  --data TextFiles \
  --data.train_data_path "data" \
  --out_dir out/custom_model

# 3) Chat with the model
# echo "Chat with the model"
# litgpt chat out/custom_model/final

# 4) Deploy the model
# litgpt serve out/custom_model/final
