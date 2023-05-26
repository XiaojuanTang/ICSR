srun -J eval -p gpu -c 64 --time=24:00:00 --partition=gpu --account=research --qos=level2 -N 1 --gres=gpu:1 \
python eval1_with_input.py \
    --model_name_or_path logs/insert_sym_b16_2 \
    --data_path dataset/part0.bin \
    --use_sym True \
--output_dir .result &&
srun -J eval -p gpu -c 64 --time=24:00:00 --partition=gpu --account=research --qos=level2 -N 1 --gres=gpu:1 \
python eval1_with_input.py \
    --model_name_or_path logs/insert_sym_b16_2 \
    --data_path dataset/part1.bin \
    --use_sym True \
--output_dir .result &&
srun -J eval -p gpu -c 64 --time=24:00:00 --partition=gpu --account=research --qos=level2 -N 1 --gres=gpu:1 \
python eval2_with_input.py \
    --model_name_or_path logs/edit_sym_b16_2 \
    --data_path dataset/part0.bin \
    --use_sym True \
--output_dir .result &&
srun -J eval -p gpu -c 64 --time=24:00:00 --partition=gpu --account=research --qos=level2 -N 1 --gres=gpu:1 \
python eval3_with_input.py \
    --model_name_or_path logs/edit_sym_b16_2 \
    --data_path dataset/part1.bin \
    --use_sym True \
--output_dir .result