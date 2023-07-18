## CODE CREATER: Nishkal Prakash (nishkal@iitkgp.ac.in)

# from minutiae_extractor import extract_minutiae_vector as emv
from pathlib import Path
# from BI import Datasets
from ROI import ROI_extractor
import cv2

if __name__ == "__main__":
    from BI import Parallel

    def get_roi(input_folder_path, output_folder_path):
        all_paths_set = set(input_folder_path.rglob('*.*'))
    
        proc_set = set(output_folder_path.rglob('*.*'))
        to_process = all_paths_set - proc_set
        ll=Parallel(debug=False)
        for doc_list in ll(ROI_extractor,to_process, 200):
            # try:
            for cropped_img,path in doc_list:
                out_path=output_folder_path/path.name
                # path=Path(path)
                ## check if shape of cropped image is not 0
                if cropped_img.shape==0 or cropped_img.shape[0]==0 or cropped_img.shape[1]==0:
                    print(f"Failed for {path.parent.parent/path.name}")
                else:
                    cv2.imwrite(out_path.as_posix(),cropped_img)

                
    input_paths=[
        Path(r"Y:\FVC Fingerprint Datasets\FVC2006\Dbs\DB2_A"),
        Path(r"Y:\FVC Fingerprint Datasets\FVC2004\Dbs\DB2_A"),
    ]
    output_paths=[
        Path(r"Y:\FVC Fingerprint Datasets\FVC2006\Dbs\DB2_A_ROI"),
        Path(r"Y:\FVC Fingerprint Datasets\FVC2004\Dbs\DB2_A_ROI")
    ]
    ## create output paths if not exists
    for p in output_paths:
        if not p.exists():
            p.mkdir(parents=True)
    for inp, out in zip(input_paths, output_paths):
        get_roi(inp, out)




    