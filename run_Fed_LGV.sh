# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
noise_rates=(0.3)
# noise_rates=(0.4 0.5 0.6 0.7 0.8 0.9 1.0)
vuls=("reentrancy" "timestamp")
# vuls=("reentrancy")
# global_weights=(0.5 0.55 0.6 0.65 0.7 0.75)
global_weights=(0.75)
device="$1"
num_neigh="$2"

for gw in ""${global_weights[@]}
do
    for vul in "${vuls[@]}"
    do
        for noise_rate in "${noise_rates[@]}"
        do
        # 对每个噪声值运行五次
        for i in {1..10}
        do
            # python Fed_LGV.py --vul $vul --noise_type non_noise --noise_rate $noise_rate --epoch 30 --warm_up_epoch 25 --device "$device" --batch 8  --random_noise --global_weight 0.75 --num_neigh "$num_neigh" --lab_name Fed_LGV --model_type CBGRU --diff --consistency_score
            python Fed_LGV.py --vul $vul --noise_type non_noise --noise_rate $noise_rate --epoch 30 --warm_up_epoch 25 --batch 8  --random_noise --global_weight 0.75 --num_neigh "$num_neigh" --lab_name Fed_LGV --model_type CBGRU --diff --consistency_score
            # python Fed_LGV.py --vul $vul --noise_type sys_noise --noise_rate $noise_rate --epoch 45 --warm_up_epoch 30 --device "$device" --batch 4  --random_noise --global_weight 0.65 --lab_name new_feature_Fed_LGV --model_type CGE 
            # python Fed_LGV.py --vul $vul --noise_type non_noise --noise_rate $noise_rate --epoch 25  --warm_up_epoch 50 --device "$device"  --random_noise --global_weight "$gw"  --lab_name label_Fed_LGV --model_type CGE 
        done
        done
    done
done
