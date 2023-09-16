import os
import argparse
import numpy as np
import cv2
import shutil
import re
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='my_313', type=str)
parser.add_argument('--output_dir', default='my_313', type=str)
parser.add_argument('--type', default='annots', type=str)
args = parser.parse_args()

def save_cams(cams, num_cams, dest_root):
    # Create FileStorage objects in write mode
    intri = cv2.FileStorage(os.path.join(dest_root, 'intri.yml'), cv2.FILE_STORAGE_WRITE)
    extri = cv2.FileStorage(os.path.join(dest_root, 'extri.yml'), cv2.FILE_STORAGE_WRITE)

    # Save intrinsic parameters
    for i in num_cams:
        intri.write('K_{}'.format(i), cams['K'][i])
        intri.write('dist_{}'.format(i), cams['D'][i].T)

    # Save extrinsic parameters
    for i in num_cams:
        extri.write('Rot_{}'.format(i), cams['R'][i])
        extri.write('T_{}'.format(i), cams['T'][i] / 1000)  # Assuming you want to reverse the scaling by 1000

    # Release the FileStorage objects
    intri.release()
    extri.release()

def copy_images(all_ims, input_dir, output_dir):
    for frame_idx, frame in enumerate(all_ims):
        for img_path in frame:
            # Parse the camera index from the original image path
            match = re.search(r'Camera_B(\d+)', img_path)
            if match:
                cam_idx = match.group(1)  # Extract the camera index without 'B'
                
                # Create the destination folder
                dest_folder = os.path.join(output_dir, 'images', cam_idx)
                os.makedirs(dest_folder, exist_ok=True)
                
                # Create the destination path
                dest_path = os.path.join(dest_folder, os.path.basename(img_path))
                
                source_path = os.path.join(input_dir, img_path)
                # Copy the image to the destination folder
                shutil.copy(source_path, dest_path)

if args.type == 'annots':
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    anno_path = os.path.join(args.input_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()

    # cams
    # cams = {'K': [], 'D': [], 'R': [], 'T': []}
    cams = annots['cams']
    num_cams = [i for i in range(len(cams['K']))]
    save_cams(cams, num_cams, output_dir)

    # load image paths
    img_paths_per_frame = [d['ims'] for d in annots['ims']]
    copy_images(img_paths_per_frame, input_dir=args.input_dir, output_dir=output_dir)
elif args.type == 'vertices':
    param_out = os.path.join(args.output_dir, 'output-smpl-3d/smpl/')
    os.makedirs(param_out, exist_ok=True)

    param_in = sorted(os.listdir(os.path.join(args.input_dir, 'params')), key=lambda x: int(os.path.splitext(x)[0]))
    vert_in = sorted(os.listdir(os.path.join(args.input_dir, 'vertices')), key=lambda x: int(os.path.splitext(x)[0]))
    assert len(param_in) == len(vert_in)

    for idx, (param_path, vert_path) in enumerate(zip(param_in, vert_in)):
        # Load the params and vertices from the .npy files
        params = np.load(os.path.join(args.input_dir, 'params', param_path), allow_pickle=True).item()
        vertices = np.load(os.path.join(args.input_dir, 'vertices', vert_path))

        # Extract the individual parameters
        poses = params['poses']
        Rh = params['Rh']
        Th = params['Th']
        shapes = params['shapes']

        # Reconstruct the original params dictionary
        # single person, so
        reconstructed_params = {
            'annots': [
                {
                    'id': 0,
                    'shapes': shapes.tolist(),
                    'poses': poses.tolist(),
                    'Rh': Rh.tolist(),
                    'Th': Th.tolist(),
                }
            ]
        }

        # Save the reconstructed params as a JSON file
        param_in_full_reconstructed = os.path.join(param_out, "{:0>6}.json".format(idx))
        with open(param_in_full_reconstructed, 'w') as f:
            json.dump(reconstructed_params, f)
