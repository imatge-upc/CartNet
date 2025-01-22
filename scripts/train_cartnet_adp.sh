
cd ..
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --name "CartNet_2" --model "CartNet" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment &
CUDA_VISIBLE_DEVICES=1 python main.py --seed 1 --name "CartNet_2" --model "CartNet" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment &
CUDA_VISIBLE_DEVICES=2 python main.py --seed 2 --name "CartNet_2" --model "CartNet" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment &
CUDA_VISIBLE_DEVICES=3 python main.py --seed 3 --name "CartNet_2" --model "CartNet" --dataset "ADP" --dataset_path "/scratch/g1alexs/ADP_DATASET"  \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50 \
                                    --augment &
                                  
wait

python test_metrics_adp.py --path "./results/CartNet_2"