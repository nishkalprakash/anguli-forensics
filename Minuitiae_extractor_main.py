from BI import Parallel
from minutiae_extractor import extract_minutiae_vector as emv

if __name__ == "__main__":
    from pathlib import Path
    from pymongo import MongoClient as mc

    coll='Anguli_test'
    base='Anguli_test'
    ext='*.tiff'
    db_client = mc("10.5.18.101")["BI"][coll]
    db_client.create_index('path')
    all_paths_set_file=Path(f"{base}/{base}_file_list.txt")
    if all_paths_set_file.exists():
        all_paths_set = set(Path(all_paths_set_file).read_text().split('\n'))
    else:
        def create_all_paths(base="Anguli_200k_1M",ext="*.tiff"):
            images = Path(base).rglob(ext)
            all_paths = map(str,sorted(images))
            images_str = "\n".join(all_paths)
            Path(all_paths_set_file).write_text(images_str)
            return set(all_paths)
        all_paths_set = create_all_paths(base,ext)
    proc_set = {i['path'] for i in db_client.find({},{'path':1,'_id':0})}
    to_process = all_paths_set - proc_set
    # for p in to_process:
    #     print(db_client.insert_one(emv(p)))

    ll=Parallel()
    for doc_list in ll(emv,to_process,100,10,1):
        try:
            result=db_client.insert_many(doc_list,ordered=False)
        except:
            print(f"Inserted - {len(result.inserted_ids)}")

    # import re
    # """
    # finger=4
    # cur=list(db_client.find({'path':re.compile('Anguli_200k_1M/Impressions/Impression_\d/fp_1/finger.tiff')}))
    # from pprint import pprint
    # with Path('finger_minutiae.txt').open('w+') as f:
    #     pprint(cur,f)
    # """
