
EVAL_PATH="/content/drive/MyDrive/BDMP_update"
GT_NAME="XiaoxiChen"
PR_NAME='/content/drive/MyDrive/BDMP_update/BDMAP_00000031'
REFERENCE_CT_PATH="/content/drive/MyDrive/BDMP/BDMAP_00000031/ct.nii.gz" 
NUM_CLASSES=24 

python eval.py --evalpath $EVAL_PATH --gt $GT_NAME --pr $PR_NAME --reference_ct $REFERENCE_CT_PATH --num_classes $NUM_CLASSES
