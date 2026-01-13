# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
# noise_rates=(0.3)
# noise_rates=(0.4 0.5 0.6 0.7 0.8 0.9 1.0)
vuls=("reentrancy" "timestamp")
device="$1"

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行五次
    for i in {1..10}
    do
        # python Fed_LGV.py --vul $vul --noise_type sys_noise --noise_rate $noise_rate --epoch 30 --warm_up_epoch 25 --device "$device" --batch 4  --random_noise --global_weight 0.65 --lab_name new_feature_Fed_LGV --model_type CBGRU
        python Fed_LGV.py --vul $vul --noise_type sys_noise --noise_rate $noise_rate --epoch 30 --warm_up_epoch 25 --device "$device" --batch 4  --random_noise --global_weight 0.5 --lab_name new_feature_Fed_LGV --model_type CBGRU --consistency_score
    done
    done
done