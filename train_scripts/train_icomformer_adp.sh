CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --name "icomformer" --model "icomformer" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch_size 64 --lr 0.001 --epochs 50  & 
CUDA_VISIBLE_DEVICES=4 python main.py --seed 1 --name "icomformer" --model "icomformer" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET" \
                                    --wandb_project "CartNet Paper" --batch_size 64 --lr 0.001 --epochs 50  & 

CUDA_VISIBLE_DEVICES=2 python main.py --seed 2 --name "icomformer" --model "icomformer" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET" \
                                    --wandb_project "CartNet Paper" --batch_size 64 --lr 0.001 --epochs 50  & 

CUDA_VISIBLE_DEVICES=3 python main.py --seed 3 --name "icomformer" --model "icomformer" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET" \
                                    --wandb_project "CartNet Paper" --batch_size 64 --lr 0.001 --epochs 50  & 