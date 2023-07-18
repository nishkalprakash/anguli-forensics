#%% 
from BI import Datasets

# load FVC2002 DB
ds=Datasets("BI")

cur=ds.connect("FVC2002_DB1")

cur.count_documents({})
# %%
docs=cur.find({})
# %%
from pathlib import Path

base=Path("FVC2002_DB1")
base.mkdir(exist_ok=True,parents=True)
#%%
for doc in docs:
    new_file=base/(doc["path"].split("/")[-1]).replace(".tif",".txt")
    data=[['X','Y','A']]
    for m in doc["mv"]:
        data.append([m[0],m[1],m[3]])
    new_file.write_text("\n".join([",".join(map(str,i)) for i in data]))
    

    # print(doc)