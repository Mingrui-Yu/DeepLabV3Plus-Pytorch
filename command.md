## Training

```
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --dataset dlo --data_root ../dlo_dataset  --train_data train/train --test_data test/test  --crop_val --lr 0.01 --crop_size 513 --batch_size 8 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_dlo_os16.pth --continue_training
```


```
python predict.py --input datasets/data/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg  --dataset voc --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --save_val_results_to test_results
```