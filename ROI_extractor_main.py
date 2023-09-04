## CODE CREATER: Nishkal Prakash (nishkal@iitkgp.ac.in) & Rakesh Krishna (rakhikrish1319@gmail.com)
## Steps to reproduce
## run in terminal
## conda create -n roi python -y
## conda activate roi
## pip install -r requirements_roi.txt


# from minutiae_extractor import extract_minutiae_vector as emv
from pathlib import Path
# from BI import Datasets
from ROI import ROI_extractor
import cv2


if __name__ == "__main__":
    from BI import Parallel

    def get_roi(input_folder_path, output_folder_path):
        all_img_names_set = set([file.name for file in input_folder_path.rglob('*.*') if file.suffix in ['.tif','.bmp','.png']])
    
        proc_set = set([file.name for file in output_folder_path.rglob('*.*')])
        to_process = [input_folder_path/path for path in (all_img_names_set - proc_set)]
        print(f"Total files to process: {len(to_process)} out of {len(all_img_names_set)}")
        ll=Parallel(debug=False)
        for doc_list in ll(ROI_extractor,to_process, 50):
            # try:
            for cropped_img,path in doc_list:
                out_path=output_folder_path/path.name
                # path=Path(path)
                ## check if shape of cropped image is not 0
                try:
                    if cropped_img.shape==0 or cropped_img.shape[0]==0 or cropped_img.shape[1]==0:
                        print(f"Failed for {path.parent.parent/path.name}")
                    else:
                        cv2.imwrite(out_path.as_posix(),cropped_img)
                        print('.',end='')
                except:
                    print(f"Failed for {path.parent.parent/path.name}")
                # if cropped_img.shape==0 or cropped_img.shape[0]==0 or cropped_img.shape[1]==0:
                #     print(f"Failed for {path.parent.parent/path.name}")
                # else:
                #     cv2.imwrite(out_path.as_posix(),cropped_img)
                #     print('.',end='')

    base_path = Path(r"D:\fvc_fingerprint_datasets")
    input_paths=[
        base_path/r"ASRA\FVC2006_DB2A_ASRA_Auto",
        base_path/r"ASRA\FVC2006_DB2B_ASRA_Auto",
        base_path/r"ASRA\FVC2006_DB3A_ASRA_Auto",
        base_path/r"ASRA\FVC2006_DB3B_ASRA_Auto",
        # base_path/"FVC2004/Dbs/DB2_A",
    ]
    output_paths=[
        base_path/r"ASRA\FVC2006_DB2A_Aligned",
        base_path/r"ASRA\FVC2006_DB2B_Aligned",
        base_path/r"ASRA\FVC2006_DB3A_Aligned",
        base_path/r"ASRA\FVC2006_DB3B_Aligned",
        # base_path/"FVC2006/Dbs/DB2_A_ROI",
        # base_path/"FVC2004/Dbs/DB2_A_ROI",
    ]
    ## create output paths if not exists
    for p in output_paths:
        if not p.exists():
            p.mkdir(parents=True,exist_ok=True)
    for inp, out in zip(input_paths, output_paths):
        get_roi(inp, out)




    