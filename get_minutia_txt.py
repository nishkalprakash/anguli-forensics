#%% 
from BI import Datasets

# load FVC2002 DB
ds=Datasets("BI")

cur=ds.connect("FVC2006")

cur.count_documents({})
# %%
query={"path":{"$regex":"DB2_A"}}
cur.count_documents(query)
# %%
docs=cur.find(query,projection={"path":1,"mv_25":1,"_id":0})
# paths=sorted([doc["path"] for doc in cur.find(query)])
# %%

# len(paths)
# %%
from pathlib import Path
home=Path("D:/FVC Fingerprint Datasets/FVC2006")
base=home/Path("FVC2006_DB2_A_m25")
base.mkdir(exist_ok=True,parents=True)
#%%
for doc in docs:
    new_file=base/(doc["path"].split("/")[-1]).replace(".bmp",".txt")
    data=[['X','Y','A']]
    for m in doc["mv_25"]:
        data.append([m[0],m[1],m[3]])
    new_file.write_text("\n".join([",".join(map(str,i)) for i in data]))
    

    # print(doc)
# %%
