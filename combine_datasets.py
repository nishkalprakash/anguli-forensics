## This code will combine all FVC2006 sets into one superset renaming all images to unique names
#%% Imports
from pathlib import Path

base_path = Path(r"D:\FVC Fingerprint Datasets\FVC2006\Dbs")
dest_path = Path(r"D:\FVC Fingerprint Datasets\FVC2006\Dbs_combined")
# make dest_path folder if not exists
# dest_path.mkdir(parents=True, exist_ok=True)
#%%
def get_dbset_finger(path):
    return path.parent.name+"/"+path.name.split('_')[0]

def get_all_impressions(base,path):
    return base.rglob(path+"_*.*")
#%%
all_img_names_set = set(get_dbset_finger(file) for file in base_path.rglob('*.*') if file.suffix in ['.tif','.bmp','.png'])
proc_set = set(get_dbset_finger(file) for file in dest_path.rglob('*.*'))
to_process = all_img_names_set - proc_set
#%%
to_process=sorted(to_process)
#%%
print(f"Total files to process: {len(to_process)} out of {len(all_img_names_set)}")
#%%
list(get_all_impressions(base_path,to_process[0]))
#%%
for i, path in enumerate(to_process,1):
    for imp in get_all_impressions(base_path,path):
        imp.rename(dest_path/f"{i}_{imp.name.split('_')[1]}")
    # path.rename(base_path/f"{i}.png")
    print('.',end='')
    if i%100==0:
        print(f"\n{i}",end='')
print(f"\n{i}",end='')
