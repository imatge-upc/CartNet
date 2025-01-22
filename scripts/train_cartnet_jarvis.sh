
cd ..
echo "--figshare_target can be any of this for Jarvis dataset: [optb88vdw_total_energy, optb88vdw_bandgap, mbj_bandgap, formation_energy_peratom, ehull]"

CUDA_VISIBLE_DEVICES=0 python main.py --seed 1 --figshare_target "formation_energy_peratom" --name "jarvis_dft_3D_formation_energy_peratom" --model "CartNet" --dataset "jarvis" --dataset_path "./dataset/jarvis/"  \
                                    --wandb_project "CartNet Paper Jarvis" --batch 64 --batch_accumulation 1 --lr 0.001 --epochs 500  &
CUDA_VISIBLE_DEVICES=1 python main.py --seed 2 --figshare_target "formation_energy_peratom" --name "jarvis_dft_3D_formation_energy_peratom" --model "CartNet" --dataset "jarvis" --dataset_path "./dataset/jarvis/"  \
                                    --wandb_project "CartNet Paper Jarvis" --batch 64 --batch_accumulation 1 --lr 0.001 --epochs 500  &                                   
CUDA_VISIBLE_DEVICES=2 python main.py --seed 3 --figshare_target "formation_energy_peratom" --name "jarvis_dft_3D_formation_energy_peratom" --model "CartNet" --dataset "jarvis" --dataset_path "./dataset/jarvis/"  \
                                    --wandb_project "CartNet Paper Jarvis" --batch 64 --batch_accumulation 1 --lr 0.001 --epochs 500  &
CUDA_VISIBLE_DEVICES=3 python main.py --seed 4 --figshare_target "formation_energy_peratom" --name "jarvis_dft_3D_formation_energy_peratom" --model "CartNet" --dataset "jarvis" --dataset_path "./dataset/jarvis/"  \
                                    --wandb_project "CartNet Paper Jarvis" --batch 64 --batch_accumulation 1 --lr 0.001 --epochs 500  &


wait

python test_metrics.py --path "./results/jarvis_dft_3D_formation_energy_peratom"