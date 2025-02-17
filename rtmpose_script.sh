# RTMDet 与 RTMPose 联合推理
# 输入模型路径可以是本地路径，也可以是下载链接。
python demo/topdown_demo_with_mmdet.py \
    projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    /home/featurize/app/weights/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    /home/featurize/app/weights/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input /home/featurize/work/mmpose/1261739776249_.pic.jpg \
    --output-root vis_case.jpg \
    --draw-bbox