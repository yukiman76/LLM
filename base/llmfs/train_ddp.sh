export MASTER_ADDR='localhost'
export MASTER_PORT='12355'
export OMP_NUM_THREADS='1'
# torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 \
#          --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#          llfs_train_ddp.py

torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100  \
         --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         llfs_train_ddp.py 

# torchrun --standalone 
# --standalone

# torchrun \
# --nproc_per_node=2 --nnodes=1 --node_rank=0 \
# --master_addr=127.0.0.1 --master_port=12355 \
# llfs_train_ddp.py
