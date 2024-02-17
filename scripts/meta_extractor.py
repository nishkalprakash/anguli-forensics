## CODE CREATER: Nishkal Prakash (nishkal@iitkgp.ac.in)
## Code to read meta files and add them to mongodb

# from minutiae_extractor import extract_minutiae_vector as emv
from pathlib import Path
from BI import Datasets

def extract_meta_dict(meta_path):
    """
        Reads meta files and adds them to mongodb

        Schema of meta db:
        {
            'meta_path': `meta_file_path`(str), 
            'path': `master_fingerprint_path`(str), 
            'type': `fingerprint_type(whorl|double_whorl|left_loop|right_loop|arch|tented_arch)`, 
            ('loop ': [
                {
                    'x': `x_coord`(int), 
                    'y': `y_coord`(int), 
                }
            ]`len(1|2)`,
            'delta': [
                {
                    'x': `x_coord`(int), 
                    'y': `y_coord`(int), 
                }
            ]`len(1|2)`,)`not available for arch`
            ('arch':`arching amount`(float))`only available for arch`
        }
    """
    meta_dict={
        'meta_path': str(meta_path),
        'path': str(meta_path).replace('Meta Info','Fingerprints').replace('.txt','.tiff')
        }
    for line in Path(meta_path).read_text().lower().strip().split('\n'):
        kv=line.split(' : ')
        try:
            try:
                x,y=map(int,kv[1].strip().split())
                d=dict(x=x,y=y)
                try:
                    meta_dict[kv[0]].append(d)
                except KeyError:
                    meta_dict[kv[0]]=[d]
            except ValueError:
                meta_dict[kv[0]]=kv[1].strip().replace(' ','_')
        except IndexError:
            meta_dict['arch']=float(kv[0])
    return meta_dict

# extract_meta_dict("Anguli_200k_1M/Meta Info/fp_1/4.txt")
if __name__ == "__main__":
    from pymongo import MongoClient as mc
    # prefix = 
    from BI import Datasets, Parallel

    dsn = "Anguli_200k_1M"
    ds=Datasets()
    coll=ds.connect(dsn)
    
    meta_path=ds.path/"Meta Info"
    all_paths_set = set(meta_path.rglob("*.txt"))
    
    # def create_all_paths():
        # meta_paths = Path("Anguli_200k_1M/Meta Info").rglob("*.txt")
        # meta_paths_str = "\n".join(map(str,sorted(meta_paths)))
        # Path("Anguli_200k_1M/MetaData_file_list.txt").write_text(meta_paths_str)    

    proc_set = {i.get('meta_path','') for i in coll.find({},{'meta_path':1,'_id':0})}
    to_process = all_paths_set - proc_set
    ll=Parallel(debug=True)
    for doc_list in ll(extract_meta_dict,to_process, 200):
        # try:
        for doc in doc_list:
            coll.update_one({'path':doc['path']},{'$set':doc})
        # except Exception as e:
            # print(str(e))