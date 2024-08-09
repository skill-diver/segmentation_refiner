# import os
# import nibabel as nib
# import numpy as np
# import subprocess
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import gc

# def load_and_check_nifti_file(filepath):
#     img = nib.load(filepath)
#     data = img.get_fdata()
#     if np.count_nonzero(data) == 0:
#         return None
#     return img, data

# def combine_vertebrae_labels(label_path, output_path):
#     combined_data = None
#     combined_affine = None

#     for filename in os.listdir(label_path):
#         if filename.startswith(('vertebrae_C', 'vertebrae_L', 'vertebrae_T')) and filename.endswith('.nii.gz'):
#             filepath = os.path.join(label_path, filename)
#             result = load_and_check_nifti_file(filepath)
#             if result is None:
#                 print(f"Skipping empty file: {filename}")
#                 continue

#             img, data = result
#             if combined_data is None:
#                 combined_data = np.zeros_like(data, dtype=np.float32)
#                 combined_affine = img.affine

#             combined_data += data

#     if combined_data is not None:
#         combined_img = nib.Nifti1Image(combined_data, combined_affine)
#         nib.save(combined_img, output_path)
#         print(f"Combined label saved to {output_path}")
#     else:
#         print("No valid vertebrae labels found.")


# def execute_command(idx, base_path):
#     try:
#         dir_name = f"BDMAP_{idx:08d}"
#         label_path = os.path.join(base_path, dir_name, "segmentations")
#         output_path = os.path.join(base_path, dir_name, "combined_vertebrae_label.nii.gz")

#         # Log the current directory being processed
#         print(f"\n-------- Processing {dir_name} --------\n")

#         # Combine vertebrae labels
#         combine_vertebrae_labels(label_path, output_path)

#         # Construct command for test.py
#         ct_file = os.path.join(base_path, dir_name, "ct.nii.gz")
#         command = [
#             "python", "test.py",
#             "-D", ct_file,
#             "-B", output_path,
#             "-P", label_path,
#             "-S", os.path.join(base_path, dir_name),
#         ]
#         print(f"Executing command: {' '.join(command)}")
#         result = subprocess.run(command, capture_output=True, text=True, check=True)
#         print(f"Output:\n{result.stdout}")
#         print(f"Errors:\n{result.stderr}")
#     except subprocess.CalledProcessError as e:
#         print(f"Command '{' '.join(command)}' failed with return code {e.returncode}")
#         print(f"Output:\n{e.output}")
#         print(f"Errors:\n{e.stderr}")
#     except Exception as e:
#         print(f"Exception for index {idx}: {e}")
#     finally:
#         gc.collect()
#     # Log completion of the current directory
#     print(f"\n-------- Finished {dir_name} --------\n")

# def main():
#     base_path = "/ccvl/net/ccvl15/jingxing/AbdomenAtlasPro_update"
#     start_idx = 1
#     end_idx = 5500

#     indices = list(range(start_idx, end_idx + 1))
#     max_workers = 1  

#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = {executor.submit(execute_command, idx, base_path): idx for idx in indices}
#         for future in as_completed(futures):
#             idx = futures[future]
#             try:
#                 future.result()
#             except Exception as exc:
#                 print(f"Command for index {idx} generated an exception: {exc}")

# if __name__ == "__main__":
#     main()

# import os
# import nibabel as nib
# import numpy as np
# import subprocess
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import gc

# def load_and_check_nifti_file(filepath):
#     img = nib.load(filepath)
#     data = img.get_fdata()
#     if np.count_nonzero(data) == 0:
#         return None
#     return img, data

# def combine_vertebrae_labels(label_path, output_path):
#     combined_data = None
#     combined_affine = None

#     for filename in os.listdir(label_path):
#         if filename.startswith(('vertebrae_C', 'vertebrae_L', 'vertebrae_T')) and filename.endswith('.nii.gz'):
#             filepath = os.path.join(label_path, filename)
#             result = load_and_check_nifti_file(filepath)
#             if result is None:
#                 print(f"Skipping empty file: {filename}")
#                 continue

#             img, data = result
#             if combined_data is None:
#                 combined_data = np.zeros_like(data, dtype=np.float32)
#                 combined_affine = img.affine

#             combined_data += data

#     if combined_data is not None:
#         combined_img = nib.Nifti1Image(combined_data, combined_affine)
#         nib.save(combined_img, output_path)
#         print(f"Combined label saved to {output_path}")
#     else:
#         print("No valid vertebrae labels found.")

# def execute_command(idx, base_path, ct_path, gpu_id):
#     try:
#         dir_name = f"BDMAP_{idx:08d}"
#         label_path = os.path.join(base_path, dir_name, "segmentations")
#         output_path = os.path.join(base_path, dir_name, "combined_vertebrae_label.nii.gz")

#         # Log the current directory being processed
#         print(f"\n-------- Processing {dir_name} on GPU {gpu_id} --------\n")

#         # Combine vertebrae labels
#         combine_vertebrae_labels(label_path, output_path)

#         # Construct command for test.py
#         ct_file = os.path.join(ct_path, dir_name, "ct.nii.gz")
#         command = [
#             "python", "test.py",
#             "-D", ct_file,
#             "-B", output_path,
#             "-P", label_path,
#             "-S", os.path.join(base_path, dir_name),
#         ]
#         print(f"Executing command: {' '.join(command)} on GPU {gpu_id}")
#         env = os.environ.copy()
#         env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#         result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
#         print(f"Output:\n{result.stdout}")
#         print(f"Errors:\n{result.stderr}")
#     except subprocess.CalledProcessError as e:
#         print(f"Command '{' '.join(command)}' failed with return code {e.returncode}")
#         print(f"Output:\n{e.output}")
#         print(f"Errors:\n{e.stderr}")
#     except Exception as e:
#         print(f"Exception for index {idx}: {e}")
#     finally:
#         gc.collect()
#     # Log completion of the current directory
#     print(f"\n-------- Finished {dir_name} --------\n")

# def main():
#     base_path = "/ccvl/net/ccvl15/jingxing/AbdomenAtlasPro_update"
#     ct_path= "/mnt/T9/AbdomenAtlasPro"
#     start_idx = 1
#     end_idx = 5500
#     num_gpus = 4

#     indices = list(range(start_idx, end_idx + 1))
#     grouped_indices = [indices[i:i + num_gpus] for i in range(0, len(indices), num_gpus)]

#     with ProcessPoolExecutor(max_workers=num_gpus) as executor:
#         futures = []
#         for group in grouped_indices:
#             for i, idx in enumerate(group):
#                 gpu_id = i % num_gpus
#                 futures.append(executor.submit(execute_command, idx, base_path, ct_path, gpu_id))

#         for future in as_completed(futures):
#             try:
#                 future.result()
#             except Exception as exc:
#                 print(f"Command generated an exception: {exc}")

# if __name__ == "__main__":
#     main()
import os
import nibabel as nib
import numpy as np
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

def load_and_check_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    if np.count_nonzero(data) == 0:
        return None
    return img, data

def combine_vertebrae_labels(label_path, output_path):
    combined_data = None
    combined_affine = None

    for filename in os.listdir(label_path):
        if filename.startswith(('vertebrae_C', 'vertebrae_L', 'vertebrae_T')) and filename.endswith('.nii.gz'):
            filepath = os.path.join(label_path, filename)
            result = load_and_check_nifti_file(filepath)
            if result is None:
                print(f"Skipping empty file: {filename}")
                continue

            img, data = result
            if combined_data is None:
                combined_data = np.zeros_like(data, dtype=np.float32)
                combined_affine = img.affine

            combined_data += data

    if combined_data is not None:
        combined_img = nib.Nifti1Image(combined_data, combined_affine)
        nib.save(combined_img, output_path)
        print(f"Combined label saved to {output_path}")
    else:
        print("No valid vertebrae labels found.")

def execute_command(idx, base_path, ct_path, gpu_id):
    try:
        dir_name = f"BDMAP_{idx:08d}"
        label_path = os.path.join(base_path, dir_name, "segmentations")
        output_path = os.path.join(base_path, dir_name, "combined_vertebrae_label.nii.gz")

        print(f"\n-------- Processing {dir_name} on GPU {gpu_id} --------\n")

        combine_vertebrae_labels(label_path, output_path)

        ct_file = os.path.join(ct_path, dir_name, "ct.nii.gz")
        command = [
            "python", "test.py",
            "-D", ct_file,
            "-B", output_path,
            "-P", label_path,
            "-S", os.path.join(base_path, dir_name),
        ]
        print(f"Executing command: {' '.join(command)} on GPU {gpu_id}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
        print(f"Output:\n{result.stdout}")
        print(f"Errors:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Command '{' '.join(command)}' failed with return code {e.returncode}")
        print(f"Output:\n{e.output}")
        print(f"Errors:\n{e.stderr}")
    except Exception as e:
        print(f"Exception for index {idx}: {e}")
    finally:
        gc.collect()
    print(f"\n-------- Finished {dir_name} --------\n")

def main():
    base_path = "/ccvl/net/ccvl15/jingxing/AbdomenAtlasPro_update"
    ct_path= "/mnt/T9/AbdomenAtlasPro"
    start_idx = 5000
    end_idx = 10000
    num_gpus = 4

    indices = list(range(start_idx, end_idx + 1))

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(execute_command, idx, base_path, ct_path, i % num_gpus): idx for i, idx in enumerate(indices)}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Command for index {idx} generated an exception: {exc}")

if __name__ == "__main__":
    main()
