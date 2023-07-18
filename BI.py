# class Batcher:
#     def __init__(self,
#                     folder='/home/rs/19CS91R05/DarKSkuLL/BI/Anguli_200k_1M',
#                     ext='*.tiff',
#                     batch_size=1000
#                 ) -> iter:
#         self.folder=folder
#         self.ext=ext
#         self.batch_size=1000
#         self.proc_count=cpu_count()-1
#         self.batch = iter()

#     # def
#     def __call__(self, ) -> iter:
#         return next(self.batch)

from multiprocessing import Pool
from os import cpu_count
from time import strftime
from vars import *

def dprint(func):
    def wrapped_func(*args, **kwargs):
        return func(strftime("%H:%M:%S - "),*args,**kwargs)
    return wrapped_func

print = dprint(print)

class Parallel:
    def __init__(self,debug=False) -> object:
        self.debug=debug
        print("Parallel Instance Created")
    
    def batcher(self,doc):
        if doc:
            self.doc_list.append(doc)
            self.ctr += 1
            self.done += 1
            if self.ctr == self.batch_size:
                self.ctr = 0
                print(f"{self.done}/{len(self.paths)} documents processed")
                dc=self.doc_list[:]
                self.doc_list = []
                return dc
        
            if len(self.doc_list):
                return self.doc_list

    def __call__(
        self, atomic_function, paths, batch_size=100, chunksize=10, free_core=1
    ) -> dict:
        self.paths=paths
        self.batch_size=batch_size
        self.ctr = self.done = 0
        self.doc_list = []
        if not self.debug:
            with Pool(cpu_count() - free_core - 1) as p:
                for doc in p.imap(atomic_function, paths, chunksize):
                    yield self.batcher(doc)                        
        else:
            for path in paths:
                yield self.batcher(atomic_function(path))

from pymongo import MongoClient, collection

class Datasets:
    """
    In: Name of Dataset
    Out: Obj
    param: 
        path - Absolute path of the ds in disk
        dbcoll - name of the db collection
        avail_ds - List of all ds avail
    """
    
    def __init__(self,ds_name=None) -> object:
        
        self.avail_ds=sorted(AVAIL_DS)
        # if ds_name is not None:
            # self.connect(ds_name=ds_name)
    
    def connect(self,ds_name,coll=None) -> collection:
        """Will connect to the database into a given collection"""
        if 1 or ds_name in self.avail_ds:
            self.ds_name=ds_name if coll is None else coll
            self.ds_prefix=Path(DS_PREFIX)
            self.ds_path=self.ds_prefix/self.ds_name
            # self.dbcol=self.ds_name if coll is None else coll
            # return self
        else:
            print(ds_name,"Collection Not found in available datasets.")
        
        
        self.con=MongoClient(DB_IP)
        return self.con[DB_NAME][self.ds_name]
        # return self.conn

    def __del__(self):
        try:
            self.con.close()
        except Exception as e:
            print(str(e))
            
