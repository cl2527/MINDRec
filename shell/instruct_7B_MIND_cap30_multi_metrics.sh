echo $1, $2
seed=$2
output_dir='results/MIND_30cap/tra_NU10_val_NU1_te_NU1_histLen_30/re'
base_model='baffo32/decapoda-research-llama-7B-hf'
train_data='data/MIND_30cap/tra_NU10_val_NU1_te_NU1_histLen_30/train.json'
val_data='data/MIND_30cap/tra_NU10_val_NU1_te_NU1_histLen_30/valid.json'
instruction_model=XXX
for lr in 1e-4;
do
    for dropout in 0.05;
    do
        for sample in 1024 2048 4096;
        do 
                mkdir -p $output_dir
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
                CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec_MIND.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --val_data_path $val_data \
                    --output_dir ${output_dir}_${seed}_${sample}_${lr}\
                    --batch_size 128 \
                    --micro_batch_size 8 \
                    --num_epochs 10 \
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

