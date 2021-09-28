export PYTHONPATH=~/workspace/megengine-trafficsign/trafficdet:$PYTHONPATH
cd ~/workspace/megengine-trafficsign/trafficdet
python3 tools/train.py -n 4 -b 2 -f configs/atss_res50_800size_trafficdet_demo.py -d ./dataset/dataset-2805 -w weights/atss_res50_coco_3x_800size_42dot6_9a92ed8c.pkl
