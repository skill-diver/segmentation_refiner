# import os
# import argparse
# import numpy as np
# import nibabel as nib
# from skimage.transform import resize

# def resize_mask(mask, target_shape):
#     return resize(mask, target_shape, order=0, preserve_range=True, anti_aliasing=False)

# vertebrae_labels = {
#     "vertebrae_C1": 1,
#     "vertebrae_C2": 2,
#     "vertebrae_C3": 3,
#     "vertebrae_C4": 4,
#     "vertebrae_C5": 5,
#     "vertebrae_C6": 6,
#     "vertebrae_C7": 7,
#     "vertebrae_T1": 8,
#     "vertebrae_T2": 9,
#     "vertebrae_T3": 10,
#     "vertebrae_T4": 11,
#     "vertebrae_T5": 12,
#     "vertebrae_T6": 13,
#     "vertebrae_T7": 14,
#     "vertebrae_T8": 15,
#     "vertebrae_T9": 16,
#     "vertebrae_T10": 17,
#     "vertebrae_T11": 18,
#     "vertebrae_T12": 19,
#     "vertebrae_L1": 20,
#     "vertebrae_L2": 21,
#     "vertebrae_L3": 22,
#     "vertebrae_L4": 23,
#     "vertebrae_L5": 24,
# }

# def get_key(my_dict, val):
#     for key, value in my_dict.items():
#         if val == value:
#             return key
#     return "key doesn't exist"

# def dice_coefficient(y_true, y_pred):
#     intersection = np.sum(y_true * y_pred)
#     return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

# def get_size_and_spacing_and_orientation_from_nifti_file(file):
#     data = nib.load(file)
#     size = data.shape
#     a, b, c = nib.orientations.aff2axcodes(data.affine)
#     orientation_code = a + b + c
#     header = data.header
#     pixdim = header['pixdim']
#     spacing = pixdim[1:4]
#     aff = data.affine
#     return size, spacing, orientation_code, aff

# def resampling(nifti_img, spacing, target_shape=None):
#     from nilearn.image import resample_img
#     import numpy as np

#     new_affine = np.copy(nifti_img.affine)
#     new_affine[:3, :3] *= 1.0 / spacing

#     if target_shape is None:
#         target_shape = (nifti_img.shape * spacing).astype(np.int32)

#     resampled_nifti_img = resample_img(nifti_img, target_affine=new_affine, target_shape=target_shape, interpolation='nearest')
#     return resampled_nifti_img

# def reorienting(img, start_orient_code, end_orient_code):
#     start_orient = nib.orientations.axcodes2ornt(start_orient_code)
#     end_orient = nib.orientations.axcodes2ornt(end_orient_code)
#     trans = nib.orientations.ornt_transform(start_orient, end_orient)
#     return nib.orientations.apply_orientation(img, trans)

# def process_binary_mask(mask_file, reference_ct_file):
#     mask = nib.load(mask_file).get_fdata()
#     _, ref_spacing, ref_orientation, ref_affine = get_size_and_spacing_and_orientation_from_nifti_file(reference_ct_file)
#     resampled_mask_img = resampling(nib.Nifti1Image(mask, ref_affine), ref_spacing)
#     resampled_mask_data = resampled_mask_img.get_fdata()
#     transformed_mask_data = reorienting(resampled_mask_data, ref_orientation, 'PIR')
#     return transformed_mask_data

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--evalpath', default=None, type=str, help='data directory')
#     parser.add_argument('--gt', default='XiaoxiChen', type=str, help='name of manual annotation')
#     parser.add_argument('--pr', default=None, type=str, help='name of AI annotation')
#     parser.add_argument('--reference_ct', default=None, type=str, help='path to the reference CT file')
#     parser.add_argument('--num_classes', default=24, type=int, help='number of classes')

#     args = parser.parse_args()

#     assert args.evalpath is not None
#     assert args.pr is not None
#     assert args.gt is not None
#     assert args.reference_ct is not None

#     gt_path = os.path.join(args.evalpath, args.gt, 'AbdomenAtlasDemoPredict', 'BDMAP_00000031', 'combined_labels.nii.gz')
#     gt_mask = np.array(nib.load(gt_path).get_fdata(), dtype='int8')

#     pr_path = os.path.join(args.pr, 'ct_seg.nii.gz')
#     pr_mask = np.array(nib.load(pr_path).get_fdata(), dtype='int8')
    
#     #pr_mask = resize_mask(pr_mask, gt_mask.shape)
#     dsc = []

#     print('##### Report Summary for {}'.format(args.pr))
#     print('| class | DSC |')
#     print('|:----  |:----  |')
#     for i in range(args.num_classes):
#         gt = (gt_path == i + 1)
#         pr = (pr_mask == i + 1)
#         print('| {} | {:.1f}% |'.format(get_key(vertebrae_labels, i + 1), 100.0 * dice_coefficient(gt, pr)))
#         dsc.append(100.0 * dice_coefficient(gt, pr))
#     print('| {} | {:.1f}% |'.format('average', sum(dsc) / len(dsc)))

# if __name__ == "__main__":
#     main()
'''bash
python eval.py --evalpath /Users/zongwei.zhou/Library/CloudStorage/OneDrive-JohnsHopkins/Mentoring/Advert/Interview --gt XiaoxiChen --pr BisheshworNeupane
'''

import os
import argparse
import numpy as np

import nibabel as nib

vertebrae_labels = {
    "vertebrae_C1": 1,
    "vertebrae_C2": 2,
    "vertebrae_C3": 3,
    "vertebrae_C4": 4,
    "vertebrae_C5": 5,
    "vertebrae_C6": 6,
    "vertebrae_C7": 7,
    "vertebrae_T1": 8,
    "vertebrae_T2": 9,
    "vertebrae_T3": 10,
    "vertebrae_T4": 11,
    "vertebrae_T5": 12,
    "vertebrae_T6": 13,
    "vertebrae_T7": 14,
    "vertebrae_T8": 15,
    "vertebrae_T9": 16,
    "vertebrae_T10": 17,
    "vertebrae_T11": 18,
    "vertebrae_T12": 19,
    "vertebrae_L1": 20,
    "vertebrae_L2": 21,
    "vertebrae_L3": 22,
    "vertebrae_L4": 23,
    "vertebrae_L5": 24,
}
# vertebrae_labels = {
#     "vertebrae_C1": 24,
#     "vertebrae_C2": 23,
#     "vertebrae_C3": 22,
#     "vertebrae_C4": 21,
#     "vertebrae_C5": 20,
#     "vertebrae_C6": 19,
#     "vertebrae_C7": 18,
#     "vertebrae_T1": 17,
#     "vertebrae_T2": 16,
#     "vertebrae_T3": 15,
#     "vertebrae_T4": 14,
#     "vertebrae_T5": 13,
#     "vertebrae_T6": 12,
#     "vertebrae_T7": 11,
#     "vertebrae_T8": 10,
#     "vertebrae_T9": 9,
#     "vertebrae_T10": 8,
#     "vertebrae_T11": 7,
#     "vertebrae_T12": 6,
#     "vertebrae_L1": 5,
#     "vertebrae_L2": 4,
#     "vertebrae_L3": 3,
#     "vertebrae_L4": 2,
#     "vertebrae_L5": 1,
# }

def get_key(my_dict, val):

	for key, value in my_dict.items():
		if val == value:
			return key

	return "key doesn't exist"

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def main():
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--evalpath', default=None, type=str, help='data directory')
    parser.add_argument('--gt', default='XiaoxiChen', type=str, help='name of manual annotation')
    parser.add_argument('--pr', default=None, type=str, help='name of AI annotation')
    parser.add_argument('--num_classes', default=24, type=int, help='number of classes')

    args = parser.parse_args()

    assert args.evalpath is not None
    assert args.pr is not None
    assert args.gt is not None

    #gt_path = os.path.join(args.evalpath, args.gt, 'AbdomenAtlasDemoPredict', 'BDMAP_00000031', 'combined_labels.nii.gz')
    gt_path = os.path.join(args.evalpath, args.gt, 'AbdomenAtlasDemoPredict', 'BDMAP_00000031', 'reversed_combined_labels.nii.gz')
    gt_mask = np.array(nib.load(gt_path).get_fdata(), dtype='int8')

    #pr_path = os.path.join(args.pr, 'combined_labels.nii.gz')
    pr_path = os.path.join(args.pr, 'ct_seg.nii.gz')
    pr_mask = np.array(nib.load(pr_path).get_fdata(), dtype='int8')

    dsc = []

    print('##### Report Summary for {}'.format(args.pr))
    print('| class | DSC |')
    print('|:----  |:----  |')
    for i in range(args.num_classes):
        gt = (gt_mask == i+1)
        pr = (pr_mask == i+1)
        print('| {} | {:.1f}% |'.format(get_key(vertebrae_labels, i+1), 100.0*dice_coefficient(gt, pr)))
        dsc.append(100.0*dice_coefficient(gt, pr))
    print('| {} | {:.1f}% |'.format('average', sum(dsc) / len(dsc)))

if __name__ == "__main__":
    main()