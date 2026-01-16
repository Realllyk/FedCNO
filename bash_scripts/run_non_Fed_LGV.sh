# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
# noise_rates=(0.1 0.15 0.2 0.25 0.3)
noise_rates=(0.3)
# noise_rates=(0.5 0.6 0.7 0.8 0.9 1.0)
vuls=("reentrancy" "timestamp")
device="$1"

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行五�?    for i in {1..3}
    do
        python non_Fed_LGV.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate  --device "$device" --batch 8  --random_noise --lab_name non_Fed_LGV
    done
    done
done
