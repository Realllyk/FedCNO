# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
# noise_rates=(0.3)
noise_rates=(0.5 0.6 0.7 0.8 0.9 1.0)
vuls=("reentrancy" "timestamp")


for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行五�?    for i in {1..10}
    do
        python fed_main/Fed_LGV.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate  --device cuda:6 --batch 8  --random_noise
    done
    done
done
