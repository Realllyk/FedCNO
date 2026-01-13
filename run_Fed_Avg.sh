# noise_rates=(0.05 0.1 0.15 0.2 0.25)
# noise_rates=(0.05 0.1 0.15 0.2 0.25)
noise_rates=(0.3)
# noise_rates=(0.5)
vuls=("reentrancy" "timestamp")
device="$1"

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行十次
    for i in {1..10}
    do
        python Fed_Avg.py --vul $vul --noise_type sys_noise --noise_rate $noise_rate --cbgru_net2 bigru --device "$device" --epoch 50 --random_noise --model_type CBGRU --diff --lab_name diff_Fed_Avg
        # python Fed_Avg.py --vul $vul --noise_type non_noise --noise_rate $noise_rate --cbgru_net2 bigru --device "$device" --random_noise --model_type CGE --epoch 75
    done
    done
done 