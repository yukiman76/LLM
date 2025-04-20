
python ../train.py \
  --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --peft_mode "lora" \
  --lora_r 16 \
  --lora_alpha 32 \
  --train_file "../data/test_data.jsonl" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir "./lora_llama_output" \
  --report_to "none"