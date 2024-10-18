export MASTER_ADDR='localhost'
export MASTER_PORT='12355'
export OMP_NUM_THREADS='1'
torchrun --nnodes=1 --nproc_per_node=3 --rdzv_id=100 \
         --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 \
         llfs_train_ddp.py