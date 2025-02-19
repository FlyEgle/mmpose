# yolov8 channel x2 base  no transformers
CUDA_VISIBLE_DEVICES=0
python tools/train.py /home/featurize/work/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/yolov11s_channelx2.py \
--work-dir /home/featurize/app/ckpts/yolov11_channelx2_base \
--auto-scale-lr 