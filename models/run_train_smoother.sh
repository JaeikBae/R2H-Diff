num_node=$1
shift
python3 train_smoother.py --num_workers=$num_node $@