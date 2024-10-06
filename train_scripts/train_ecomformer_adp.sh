CUDA_VISIBLE_DEVICES=2 python main.py --seed 0 --name "ecomformer" --model "ecomformer" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50  &

CUDA_VISIBLE_DEVICES=4 python main.py --seed 1 --name "ecomformer" --model "ecomformer" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET" \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50  & 

CUDA_VISIBLE_DEVICES=2 python main.py --seed 2 --name "ecomformer" --model "ecomformer" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET" \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50  & 

CUDA_VISIBLE_DEVICES=3 python main.py --seed 3 --name "ecomformer" --model "ecomformer" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET" \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50  & 