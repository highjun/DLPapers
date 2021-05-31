for t in 1 10 100 1000
do
    echo "crd_temp_$t"
    python3 train.py --desc "crd_temp_$t" --T $t --batch_size 64
done