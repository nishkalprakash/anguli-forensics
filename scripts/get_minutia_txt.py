## This code fetches documents from MongoDB and saves each record as .mnt 
## eg: FVC2006/DB2_A/100.mnt will have first line as 
##      "{file_name=100} {minutia_count=60} {width=400} {height=500}" 
## followed by 60 lines of the format "{X=100} {Y=200} {Theta=3.14159265}"
#%% 
from lib.BI import Datasets

# load FVC2002 DB
ds=Datasets("BI")

cur=ds.connect("FVC2006")

cur.count_documents({})
# %%
db="DB3_A"
threshold=20
query={"path":{"$regex":db}}
cur.count_documents(query)
# %%
docs=cur.find(query,projection={"path":1,f"mv_{threshold}":1,"_id":0})
# paths=sorted([doc["path"] for doc in cur.find(query)])
# %%

# len(paths)
# %%
from pathlib import Path
home=Path("D:/FVC Fingerprint Datasets/FVC2006")
base=home/Path(f"FVC2006_{db}_m{threshold}")
base.mkdir(exist_ok=True,parents=True)
# %%
if db == "DB2_A":
    height,width=400,560
elif db == "DB3_A":
    height,width=400,500
#%%
for doc in docs:
    fname = Path(doc["path"]).stem
    new_file=base/(fname+".mnt")

    data=[[fname],[len(doc[f"mv_{threshold}"]),height,width]]
    for m in doc[f"mv_{threshold}"]:
        data.append([m[0],m[1],m[3]])
    new_file.write_text("\n".join([" ".join(map(str,i)) for i in data]))
    

    # print(doc)
# %%
