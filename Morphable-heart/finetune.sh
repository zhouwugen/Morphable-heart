python finetune.py \
    --image_dir path/to/your/CTAdata/image \
    --mask_dir path/to/your/CTAdata/seg \
    --ids_file your/data/ids.txt \
    --pretrain_ckpt ./checkpoints/cardiac_hmr_epoch19.pth \
    --save_dir ./finetune_ckpts \
    --epochs 50 \
    --batch_size 50 \
    --lr 1e-5