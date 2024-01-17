from BI import Parallel
from minutiae_extractor_utkarsh import extract_minutiae_vector as emv

if __name__ == "__main__":
    from pathlib import Path
    from pymongo import MongoClient as mc, UpdateOne, ASCENDING

    coll='FVC2002'
    # home=Path("D:/FVC Fingerprint Datasets/")
    # base=Path("D:/FVC Fingerprint Datasets/FVC2006/Dbs/") # old path
    home=Path(r"Y:\fvc_fingerprint_datasets")
    base=home/f"{coll}/Dbs/DB1_B"
    ext='*.bmp'
    ext='*.tif'
    db_client = mc("10.5.18.101")["BI"][coll]
    try:
        db_client.create_index([('path',ASCENDING)],unique=True)
    except Exception as e:
        print(str(e))
        print("Index already exists")
    all_paths_set_file=base/f"{coll}_file_list.txt"
    if all_paths_set_file.exists():
        all_paths_set = set(Path(all_paths_set_file).read_text().split('\n'))
    else:
        def get_all_paths(base="Anguli_200k_1M",ext="*.tiff"):
            images = Path(base).rglob(ext)
            all_paths = map(Path.as_posix,sorted(images))
            # images_str = "\n".join(all_paths)
            # Path(all_paths_set_file).write_text(images_str)
            return set(all_paths)
        all_paths_set = get_all_paths(base,ext)
    
    for thres in [20,25,30]:
        proc_set = {(home/i['path']).as_posix() for i in db_client.find({ f"mv_{thres}": { "$exists": True } },{'path':1,'_id':0})}
        to_process = all_paths_set - proc_set
        print("to_process -> ",len(to_process))
        # for p in to_process:
        #     print(db_client.insert_one(emv(p)))

        ll=Parallel(debug=False)
        for doc_list in ll(emv,((p,thres) for p in to_process)):
            try:
                # for doc in doc_list:
                    # print(doc['path'])
                if len(doc_list)==0:
                    continue
                # result=db_client.insert_many(doc_list,ordered=False)
                ## update existing document with new minutiae vector using path
                result=db_client.bulk_write([UpdateOne({'path':doc['path']},{'$set':{f'mv_{thres}':doc[f'mv_{thres}']}}) for doc in doc_list])
                # print(f"Inserted - {len(result.inserted_ids)}")
            except Exception as e:
                print(str(e))
                print("Something went wrong, unable to insert")

        # import re
        # """
        # finger=4
        # cur=list(db_client.find({'path':re.compile('Anguli_200k_1M/Impressions/Impression_\d/fp_1/finger.tiff')}))
        # from pprint import pprint
        # with Path('finger_minutiae.txt').open('w+') as f:
        #     pprint(cur,f)
        # """
