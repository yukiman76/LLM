export MASTER_ADDR='localhost'
export MASTER_PORT='29400'
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 llfs_train_ddp.py