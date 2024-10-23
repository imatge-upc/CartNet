cd ..
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --name "icomformer" --model "icomformer" --dataset "ADP" --dataset_path "./dataset/ADP_DATASET"   \
                                    --wandb_project "CartNet Paper" --batch 4 --batch_accumulation 16 --lr 0.001 --epochs 50  & 
