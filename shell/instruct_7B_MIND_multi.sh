echo $1, $2
seed=$2
output_dir='results/MIND_multi/re'
base_model='baffo32/decapoda-research-llama-7B-hf'
train_data='data/MIND_multi/train.json'
val_data='data/MIND_multi/valid.json'
instruction_model=XXX
for lr in 1e-4 2e-5 5e-5;
do
    for dropout in 0.05;
    do
        for sample in 64 128 256;
        do
                mkdir -p $output_dir
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
                CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --val_data_path $val_data \
                    --output_dir ${output_dir}_${seed}_${sample}_${lr}\
                    --batch_size 128 \
                    --micro_batch_size 8 \
                    --num_epochs 200 \
                    --learning_rate $lr \
                    --cutoff_len 2048 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint $instruction_model \
                    --sample $sample \
                    --seed $2
        done
    done
done

