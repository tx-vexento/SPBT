cd /home/nfs/share/backdoor2023/backdoor/Clone/CodeBert/sh

base_model=/home/nfs/share/backdoor2023/backdoor/base_model/codebert-base
data_dir=/home/nfs/share/backdoor2023/backdoor/Clone/dataset/java

attack_ways=(IST)
poison_rates=(0.1)
triggers=(-1.1)

for attack_way in "${attack_ways[@]}"; do
for trigger in "${triggers[@]}"; do
for poison_rate in "${poison_rates[@]}"; do

output_dir=../sh/saved_models/pretrain_${attack_way}_${trigger}_${poison_rate}
mkdir -p ${output_dir}
train_filename=${data_dir}/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl
test_filename=${data_dir}/poison/${attack_way}/${trigger}_test.jsonl
dev_filename=${data_dir}/splited/test.jsonl
log=${output_dir}/train.log

cd ../code
/home/nfs/share/backdoor2023/conda/envs/invis_backdoor/bin/python run.py \
    --do_train --do_eval --do_test \
    --output_dir=${output_dir} \
    --model_type=roberta \
    --tokenizer_name=${base_model} \
    --model_name_or_path=${base_model} \
    --train_data_file=${train_filename} \
    --eval_data_file=${dev_filename} \
    --test_data_file=${test_filename} \
    --epoch 3 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 \
    2>&1 | tee ${log}
wait
#    --do_train --do_eval \
done
done
done