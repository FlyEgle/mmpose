python demo/topdown_demo_with_mmdet.py \
    projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    /home/featurize/app/weights/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    /home/featurize/work/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/yolov5s_channelx2.py \
    /home/featurize/app/ckpts/yolov5_channelx2_base/best_coco_AP_epoch_150.pth \
    --input /home/featurize/work/mmpose/1261739776249_.pic.jpg \
    --output-root vis_case.jpg \
    --draw-bbox