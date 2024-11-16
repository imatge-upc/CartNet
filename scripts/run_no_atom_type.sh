cd ..

CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --name "CartNet_no_Z" --model "CartNet" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment --disable_atom_types &
CUDA_VISIBLE_DEVICES=2 python main.py --seed 1 --name "CartNet_no_Z" --model "CartNet" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment --disable_atom_types &
CUDA_VISIBLE_DEVICES=3 python main.py --seed 2 --name "CartNet_no_Z" --model "CartNet" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment --disable_atom_types &
CUDA_VISIBLE_DEVICES=4 python main.py --seed 3 --name "CartNet_no_Z" --model "CartNet" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment --disable_atom_types &

CUDA_VISIBLE_DEVICES=5 python main.py --seed 2 --name "CartNet_nothing" --model "CartNet" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment --disable_atom_types -disable_temp &
CUDA_VISIBLE_DEVICES=6 python main.py --seed 3 --name "CartNet_nothing" --model "CartNet" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment --disable_atom_types --disable_temp &

wait

