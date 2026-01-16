noise_rates=(0.3)
# noise_rates=(0.5 0.6 0.7 0.8 0.9 1.0)
vuls=("reentrancy" "timestamp")
fracs=(0.2 0.4 0.6 0.8)
device="$1"

for i in {1..3}
do
    for vul in "${vuls[@]}"
    do
        for noise_rate in "${noise_rates[@]}"
        do
            for frac in "${fracs[@]}"
            do
                python fed_main/Fed_PLE.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate  --device "$device" --batch 8 --epoch 50 --valid_frac $frac
            done
        done
    done
done
