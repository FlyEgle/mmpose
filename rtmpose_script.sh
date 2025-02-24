# RTMDet 与 RTMPose 联合推理
# 输入模型路径可以是本地路径，也可以是下载链接。
# python demo/topdown_demo_with_mmdet.py \
#     projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
#     /home/featurize/app/weights/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
#     /home/featurize/work/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-copy.py \
#     /home/featurize/app/ckpts/best_coco_AP_epoch_40.pth \
#     --input /home/featurize/work/mmpose/1261739776249_.pic.jpg \
#     --output-root vis_case.jpg \
#     --draw-bbox


# /home/featurize/app/weights/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
# projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py


# training
# CUDA_VISIBLE_DEVICES=0
# python tools/train.py /home/featurize/work/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-copy.py \
# --work-dir /home/featurize/app/ckpts \
# --auto-scale-lr 


# yolov5 channel x2 base  no transformers
# CUDA_VISIBLE_DEVICES=0
# python tools/train.py /home/featurize/work/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/yolov5s_channelx2.py \
# --work-dir /home/featurize/app/ckpts/yolov5_channelx2_base \
# --auto-scale-lr 

# yolov8 channel x2 base  no transformers
# CUDA_VISIBLE_DEVICES=0
# python tools/train.py /home/featurize/work/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/yolov8s_channelx2.py \
# --work-dir /home/featurize/app/ckpts/yolov8_channelx2_base \
# --auto-scale-lr 


# yolov11 3 attn
# CUDA_VISIBLE_DEVICES=0
# python tools/train.py /home/featurize/work/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/yolov11s_3_attention.py \
# --work-dir /home/featurize/app/ckpts/yolov11_channelx2_3_attn \
# --auto-scale-lr 

# yolov11 1 attn
CUDA_VISIBLE_DEVICES=0
python tools/train.py /home/featurize/work/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/yolov11s_1_attention.py \
--work-dir /home/featurize/app/ckpts/yolov11_channelx2_1_attn \
--auto-scale-lr 
