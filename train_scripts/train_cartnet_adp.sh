
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --name "CartNet" --model "CartNet" --dataset "ADP"  \
                                    --wandb_project "CartNet Paper" --batch_size 64 --lr 0.001 --epochs 1 \
                                    --augment