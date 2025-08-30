
LOCAL_BATCH_SIZE=1
GRAD_ACC=40
NPROC=4
# total batchsize = LOCAL_BATCH_SIZE * GRAD_ACC * NPROC

TAG=your-tag
PORT=17566
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m torch.distributed.launch --nproc_per_node $NPROC --master-port $PORT dp_main.py \
    --model llama \
    --chkpt-dir /your/checkpoint/dir \
    --dataset-path ./dataset \
    --log-dir /your/log/dir \
    --tag $TAG \
    --reg-lambda 0 \
    --layers 32 \
    --embed-dim 4096 \
    --max-epochs 3 \
    --heads 32 \
    --n-kv-heads 32 \
    --q-forward-input fp4e2m1b \
    --q-forward-weight fp4e2m1b \
    --q-backward-input fp4e2m1b \
    --q-backward-weight fp4e2m1b \
    --q-backward-outputgrad fp4e2m1b \
    --q-scalar 1.0 \
    --grad-clipping 2.0 \
    --win-size 256 \
    --forward-svd-warmup-steps 0 \
    --forward-svd-merge-steps -1 \
    --batch-size $LOCAL_BATCH_SIZE \
    --weight-decay 0.1 \
    --lr 1.5e-4 \
    --merged-lr 1.5e-4 \
    --grad-acc 1 \
    --lr-warmup-steps 20 \
    --enable-lowbit \
    --enable-backward-svd \
    --backward-lowrank-svd 60 \
    --backward-lowrank-niter 0 \







    
    
    
    
