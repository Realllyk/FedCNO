noise_rates=(0.3)
vuls=("reentrancy" "timestamp")
device="$1"

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行十�?    for i in {1..10}
    do
        # python fed_main/Fed_Avg.py --vul $vul --noise_type sys_noise --noise_rate $noise_rate --cbgru_net2 bigru --device "$device" --epoch 75 --random_noise --model_type CGE
        python fed_main/Fed_Corr.py --vul $vul --noise_type sys_noise --noise_rate $noise_rate --cbgru_net2 bigru --device "$device" --epoch 50 --random_noise --model_type CBGRU --diff --lab_name diff_Fed_Corr
    done
    done
done
