
cd ..
echo "--figshare_target can be any of this for Jarvis dataset: [e_form, gap pbe, mbj_bandgap, bulk modulus, shear modulus]"

CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --figshare_target "e_form" --name "megnet_eform_carnet" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 64 --batch_accumulation 1 --lr 0.001 --epochs 500  


                                                                 