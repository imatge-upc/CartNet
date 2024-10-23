
cd ..
echo "--figshare_target can be any of this for Jarvis dataset: [optb88vdw_total_energy, optb88vdw_bandgap, mbj_bandgap, formation_energy_peratom, ehull]"

CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --figshare_target "optb88vdw_bandgap" --name "jarvis_dft_3D_optb88vdw_bandgap_cartnet" --model "CartNet" --dataset "jarvis" --dataset_path "./dataset/jarvis/"  \
                                    --wandb_project "CartNet Paper Jarvis" --batch 64 --batch_accumulation 1 --lr 0.001 --epochs 500  


                                                                 