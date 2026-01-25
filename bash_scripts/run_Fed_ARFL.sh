# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
# noise_rates=(0.5 0.6 0.7 0.8 0.9 1.0)
noise_rates=(0.3)
device="$1"
vuls=("reentrancy" "timestamp")

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行十�?    for i in {1..10}
    do
        python fed_main/Fed_ARFL.py --vul $vul --noise_type sys_noise --noise_rate $noise_rate --cbgru_net2 bigru --device "$device" --batch 4 --input_channel 428 --epoch 50 --model_type CBGRU --client_num 4 --sample_rate 1.0 --diff --lab_name diff_Fed_ARFL
    done
    done
done
