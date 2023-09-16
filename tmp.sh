python3 apps/postprocess/write_vertices.py data/zju_mocap_387/output-smpl-3d/smpl data/zju_mocap_387/output-smpl-3d/vertices --cfg_model config/model/smpl.yml --mode vertices

python3 apps/postprocess/render.py data/zju_mocap_387 --exp output-smpl-3d --mode instance-d0.05 --ranges 0 200 1 --model config/model/smpl.yml

python3 apps/neuralbody/demo.py --mode soccer1_6 data/soccer1_6_with_annot --gpus 0,1,