
CUDA_VISIBLE_DEVICES=0 python main_irene.py --seed 0 --name "debug_irene" --model "CartNet" --dataset "ADP"  \
                                    --wandb_project "CartNet Paper" --batch_size 64 --lr 0.001 --epochs 1 \
                                    --augment --workers 6