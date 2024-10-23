cd ..

CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --name "CartNet_invariant" --model "CartNet" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"   \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --invariant

CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --name "CartNet_no_temp" --model "CartNet" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment --disable_temp

CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --name "CartNet_no_aug" --model "CartNet" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 
                                    
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --name "CartNet_no_env" --model "CartNet" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment --disable_envelope

CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --name "CartNet_no_H" --model "CartNet" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --disable_H 
