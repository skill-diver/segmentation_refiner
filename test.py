import nibabel as nib
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import re
from utils import get_size_and_spacing_and_orientation_from_nifti_file, resampling, reorienting

def process_binary_mask(binary_mask_file, reference_ct_file):
    # load binary mask
    mask = nib.load(binary_mask_file).get_fdata()
    # ct resolution and direction
    _, ref_spacing, ref_orientation, ref_affine = get_size_and_spacing_and_orientation_from_nifti_file(reference_ct_file)
    # resampling and redirection
    resampled_mask_img = resampling(nib.Nifti1Image(mask, ref_affine), ref_spacing)
    resampled_mask_data = resampled_mask_img.get_fdata()
    transformed_mask_data = reorienting(resampled_mask_data, ref_orientation, 'PIR')  # 确保目标方向是 PIR

    return transformed_mask_data

def get_index_from_filename(filename):
    match = re.search(r'vertebrae_([CTL])(\d+)', filename)
    if match:
        region = match.group(1)
        index = int(match.group(2))
        if region == 'C':
            return index  
        elif region == 'T':
            return 7 + index  
        elif region == 'L':
            return 19 + index  
    return None  

def process_label_file(label_file, reference_ct_file):
    mask = nib.load(label_file).get_fdata()

    _, ref_spacing, ref_orientation, ref_affine = get_size_and_spacing_and_orientation_from_nifti_file(reference_ct_file)

    resampled_label_img = resampling(nib.Nifti1Image(mask, ref_affine), ref_spacing)
    resampled_label_data = resampled_label_img.get_fdata()
    transformed_label_data = reorienting(resampled_label_data, ref_orientation, 'PIR')  

    loc = center_of_mass(transformed_label_data)  # centroid
    index = get_index_from_filename(label_file)

    return {'mask': transformed_label_data, 'location': loc, 'index': index}

def load_labels_and_centroids(label_files, reference_ct_file):
    print(f"Reference CT file: {reference_ct_file}") 
    labels = []
    for label_file in label_files:
        labels.append(process_label_file(label_file, reference_ct_file))
    sorted_labels = sorted(labels, key=lambda x: x['index'])
    print("Sorted existing labels (by index):")
    for label in sorted_labels:
        print(f"Index: {label['index']}, Location: {label['location']}, Mask shape: {label['mask'].shape}")

    return sorted_labels

def load_models(seg_spine_norm=False, seg_vert_norm=False):

    if seg_spine_norm:
        model_file_seg_binary = 'models/segmentor_spine_norm.pth'
    else:
        model_file_seg_binary = 'models/segmentor_spine.pth'

    if seg_vert_norm:
        model_file_seg_idv = 'models/segmentor_vertebra_norm.pth'
    else:
        model_file_seg_idv = 'models/segmentor_vertebra.pth'

    model_file_loc_sag = 'models/locator_sagittal.pth'
    model_file_loc_cor = 'models/locator_coronal.pth'

    id_group_model_file = 'models/classifier_group.pth'
    id_cer_model_file = 'models/classifier_cervical.pth'
    id_thor_model_file = 'models/classifier_thoracic.pth'
    id_lum_model_file = 'models/classifier_lumbar.pth'


    return {'seg_binary': model_file_seg_binary, 'seg_individual': model_file_seg_idv, 
            'loc_sagittal': model_file_loc_sag, 'loc_coronal': model_file_loc_cor, 
            'id_group': id_group_model_file, 'id_cervical': id_cer_model_file, 
            'id_thoracic': id_thor_model_file, 'id_lumbar': id_lum_model_file}


if __name__ == "__main__":

    import argparse, os, glob, sys
    from utils import mkpath, read_isotropic_pir_img_from_nifti_file
    import torch

    torch.set_grad_enabled(False)


    parser = argparse.ArgumentParser(description='Run pipeline on a single CT scan.')
    parser.add_argument('-D', '--input_data', type=str, help='a CT scan or a folder of CT scans in nifti format')
    parser.add_argument('-B', '--binary_mask', type=str, help='path to the existing spine binary mask in nifti format')
    parser.add_argument('-S', '--save_folder',  default='-1', type=str, help='folder to save the results')
    parser.add_argument('-P', '--label_path', type=str, help='path to the folder containing vertebrae label files')

    parser.add_argument('-F', '--force_recompute', action='store_true', help='set True to recompute and overwrite the results')
    parser.add_argument('-L', '--initial_locations', action='store_true', help='set True to use initial location predictions')
    parser.add_argument('-Ns', '--seg_spine_norm', action='store_true', help='set True to use normalized spine segmentor')
    parser.add_argument('-Nv', '--seg_vert_norm', action='store_true', help='set True to use normalized vertebra segmentor')
    args = parser.parse_args()


    ### results saving locations
    save_folder = args.save_folder
    if save_folder != '-1':
        mkpath(save_folder)
    else:
        current_path = os.path.abspath(os.getcwd())
        save_folder = os.path.join(current_path, 'results')
        mkpath(save_folder)


    ### load trained models
    models = load_models(seg_spine_norm=args.seg_spine_norm, seg_vert_norm=args.seg_vert_norm)


    ### inputs
    scan_list = []
    if os.path.isdir(args.input_data):
        scan_list = glob.glob(os.path.join(args.input_data, '*.nii.gz'))
    elif os.path.isfile(args.input_data):
        scan_list.append(args.input_data)
    else:
        print('It is a special file (socket, FIFO, device file)')


    for scan_file in scan_list:
        ### check results existence
        scanname = os.path.split(scan_file)[-1].split('.')[0]
        print(' ... checking: ', scanname)

        if os.path.exists(os.path.join(save_folder, '{}_seg.nii.gz'.format(scanname))) and not args.force_recompute:
            sys.exit(' ... {} result exists, not overwriting '.format(scanname))

        print(' ... starting to process: ', scanname)


        # =================================================================
        # Load the CT scan 
        # =================================================================

        ### TODO: data I/O for other formats

        #try:
        pir_img = read_isotropic_pir_img_from_nifti_file(scan_file)
        # except ImageFileError:
        #     sys.exit('The input CT should be in nifti format.')

        print(' ... loaded CT volume in isotropic resolution and PIR orientation ')


        # =================================================================
        # Spine binary segmentation
        # =================================================================

        
        
        if args.binary_mask:
            #binary_mask = read_isotropic_pir_img_from_nifti_file(args.binary_mask)
            binary_mask=process_binary_mask(args.binary_mask, scan_file)
        else:
            from segment_spine import binary_segmentor
            binary_mask = binary_segmentor(pir_img, models['seg_binary'], mode='overlap', norm=args.seg_spine_norm) 

        print(' ... obtained spine binary segmentation ')

        
        # =================================================================
        # Initial locations
        # =================================================================

        locations = np.array([])
        if args.initial_locations:
            from locate import locate 
            locations = locate(pir_img, models['loc_sagittal'], models['loc_coronal'])

        print(' ... obtained {} initial 3D locations '.format(len(locations)))

        # =================================================================
        # Consistency circle - Locations refine - multi label segmentation 
        # =================================================================

        from consistency_loop import consistency_refinement_close_loop
        label_files = sorted(
            [os.path.join(args.label_path, f) for f in os.listdir(args.label_path)
            if f.startswith("vertebrae_C") or f.startswith("vertebrae_T") or f.startswith("vertebrae_L")]
        )
        # filter empty file
        #label_files = [f for f in label_files if np.count_nonzero(nib.load(f).get_fdata()) > 0]
        valid_label_files = []
        for f in label_files:
            try:
                if np.count_nonzero(nib.load(f).get_fdata()) > 0:
                    valid_label_files.append(f)
                else:
                    print(f"Skipping empty file: {f}")
            except Exception as e:
                print(f"Error loading file {f}: {e}")
                
        existing_labels =  load_labels_and_centroids(valid_label_files, scan_file)  
        
        multi_label_mask, locations, labels, loc_has_converged = consistency_refinement_close_loop(
            locations, pir_img, binary_mask, models['seg_individual'], args.seg_vert_norm,
            models['id_group'], models['id_cervical'], models['id_thoracic'], models['id_lumbar'],
            existing_labels  # exist label
        )           

        print(' ... obtained PIR multi label segmentation ')

        # =================================================================
        # Save the result in original format
        # =================================================================

        from utils import get_size_and_spacing_and_orientation_from_nifti_file, write_result_to_file
        ori_size, ori_spacing, ori_orient_code, ori_aff = get_size_and_spacing_and_orientation_from_nifti_file(scan_file)

        write_result_to_file(multi_label_mask, ori_orient_code, ori_spacing, ori_size, ori_aff, save_folder, scanname)
