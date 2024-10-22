
cd ..
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --name "jarvis_dft_3D_optb88vdw_bandgap_variantnetv4_variant_cell" --model "CartNet" --dataset "jarvis" --dataset_path "./dataset/jarvis/"  \
                                    --wandb_project "CartNet Paper" --batch 64 --batch_accumulation 1 --lr 0.001 --epochs 50 --figshare_target "optb88vdw_bandgap" \
                                    --augment


                                                                 