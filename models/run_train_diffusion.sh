num_node=$1
shift
OMP_NUM_THREADS=16 torchrun --standalone --nnodes=1 --nproc_per_node=$num_node train_diffusion.py --num_workers=$num_node $@
