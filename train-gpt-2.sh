
LOCAL_BATCH_SIZE=1
GRAD_ACC=40
NPROC=4
# total batchsize = LOCAL_BATCH_SIZE * GRAD_ACC * NPROC

TAG=your-tag
PORT=12345

python -m torch.distributed.launch --nproc_per_node $NPROC --master-port $PORT dp_main.py \
    --chkpt-dir /your/checkpoint/dir \
    --dataset-path ./dataset \
    --log-dir /your/log/dir \
    --tokenizer-path /your/tokenizer/path \
    --tag $TAG \
    --reg-lambda 0 \
    --layers 12 \
    --embed-dim 768 \
    --max-epochs 3 \
    --heads 12 \
    --n-kv-heads 12 \
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
    --grad-acc $GRAD_ACC \
    --lr-warmup-steps 20 \
    --enable-lowbit \
    --enable-backward-svd \
    --backward-lowrank-svd 60 \
    --backward-lowrank-niter 0 \







    
    
    
    
