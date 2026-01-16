noise_rates=(0.3)
vuls=("reentrancy" "timestamp")
device="$1"

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行十�?    for i in {1..10}
    do
        python fed_main/Fed_CL.py --vul $vul --noise_type sys_noise --noise_rate $noise_rate --cbgru_net2 bigru --device "$device" --random_noise --model_type CBGRU --client_num 4 --epoch 50 --diff --lab_name diff_Fed_CL
    done
    done
done
