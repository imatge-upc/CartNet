
cd ..
echo "--figshare_target can be any of this for Jarvis dataset: [e_form, gap pbe, mbj_bandgap, bulk modulus, shear modulus]"

CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --figshare_target "shear modulus" --name "megnet_shear_modulus_carnet" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 64 --batch_accumulation 1 --lr 0.001 --epochs 500 &
CUDA_VISIBLE_DEVICES=1 python main.py --seed 1 --figshare_target "shear modulus" --name "megnet_shear_modulus_carnet" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 64 --batch_accumulation 1 --lr 0.001 --epochs 500 &
CUDA_VISIBLE_DEVICES=2 python main.py --seed 2 --figshare_target "shear modulus" --name "megnet_shear_modulus_carnet" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 64 --batch_accumulation 1 --lr 0.001 --epochs 500 &
CUDA_VISIBLE_DEVICES=3 python main.py --seed 3 --figshare_target "shear modulus" --name "megnet_shear_modulus_carnet" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 64 --batch_accumulation 1 --lr 0.001 --epochs 500 &

wait

python test_metrics.py --path "./results/megnet_shear_modulus_carnet"