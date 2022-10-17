CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch --log_dir ./output_tmp train.py --feat_mode gpu --graph_mode gpu 
