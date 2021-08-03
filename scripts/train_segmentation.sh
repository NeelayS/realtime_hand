model=$1
device=$2

img_dir="segmentation/data/train_imgs"
bg_dir="segmentation/data/bg_imgs"
config_path="segmentation/seg_config.yml"
save_path="segmentation/model_weights"

cd ../
python -m segmentation.train_seg \
    --img_dir ${img_dir} \
    --bg_dir ${bg_dir} \
    --config ${config_path} \
    --model ${model} \
    --save_path ${save_path} \
    --device ${device} \
