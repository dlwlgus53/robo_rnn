CUDA_VISIBLE_DEVICES=$1,CUDA_LAUNCH_BLOCKING=1 python main.py --horizon 10 --saveto temp --expName exp1 --data norm --patience 3 --lr 0.001 --optim Adam

