# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
noise_rates=(0.3)
# noise_rates=(0.4 0.5 0.6 0.7 0.8 0.9 1.0)
vuls=("reentrancy" "timestamp")
# vuls=("reentrancy")
# global_weights=(0.5 0.55 0.6 0.65 0.7 0.75)
global_weights=(0.0)
device="$1"
num_neigh="$2"

for gw in ""${global_weights[@]}
do
    for vul in "${vuls[@]}"
    do
        for noise_rate in "${noise_rates[@]}"
        do
        # 对每个噪声值运行五�?        for i in {1..3}
        do 
            python fed_main/Fed_LGV.py --vul $vul --noise_type sys_noise --noise_rate $noise_rate --epoch 25  --warm_up_epoch 30 --device "$device"  --num_neigh "$num_neigh" --random_noise --global_weight "$gw"  --lab_name abl_no_glob --model_type CBGRU --consistency_score
        done
        done
    done
done
