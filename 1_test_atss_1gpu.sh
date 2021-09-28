export PYTHONPATH=~/workspace/megengine-trafficsign/trafficdet:$PYTHONPATH
cd ~/workspace/megengine-trafficsign/trafficdet
python3 tools/test.py -n 1 -se 23 -f configs/atss_res50_800size_trafficdet_demo.py -d ./dataset/dataset-2805
