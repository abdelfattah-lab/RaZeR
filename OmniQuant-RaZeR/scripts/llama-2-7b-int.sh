# Set your HuggingFace HOME directory to store downloaded model and datasets, default is your own HOME directory.
export HF_HOME="your/HF_HOME_directory"

# 3 bits
python main.py --model path/to/llama-2-7b-hf --epochs 20 --output_dir ./log/llama-2-7b-w3a16g128-int --eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc --lwc_lr 0.015 --aug_loss --datatype int
# 4 bits
python main.py --model path/to/llama-2-7b-hf --epochs 20 --output_dir ./log/llama-2-7b-w4a16g12-int --eval_ppl --wbits 4 --abits 16 --group_size 128 --lwc --lwc_lr 0.015 --aug_loss --datatype int