import pandas as pd
import numpy as np
import os
from typing import List

import matplotlib.pyplot as plt
import pickle
from argparse import ArgumentParser
import json
from functools import reduce
try:
    from rdkit import Chem
except ImportError:
    print("rdkit not avalable")



def is_MHC1(mhc):
    if 'HLA-A' in mhc or 'HLA-B'  in mhc or 'HLA-C'  in mhc or 'HLA-E'  in mhc or 'HLA class I'  in mhc:
        return True
    elif 'HLA-DR'  in mhc or 'HLA-DQ'  in mhc or 'HLA-DP'  in mhc:
        return False
    else:
        print(mhc)
        return False

def set_diff(df1,df2,similarity_subset=["CDR3","Epitope","alpha"]):
    """
    calculates set difference 
    df1 \ df2 
    wehre equality is based on similarity_subset
    """
   

    return df1.merge(df2[similarity_subset],on=list(similarity_subset),how='left',indicator=True).reset_index().set_index("_merge").drop("both",errors='ignore').set_index("index")

def epitope_weights(df,epitope_counts_df=None,linear=True,**kwargs):
    epi_counts = df.Epitope.value_counts() if type(epitope_counts_df) == type(None) else epitope_counts_df.Epitope.value_counts()

    if linear:
        return df.Epitope.map(lambda x : 1/epi_counts[x])
    else:
        return df.Epitope.map(lambda x : 1/(np.log(epi_counts[x])+1))

def generate_negatives(TCR_pool,Epi_pool,num_negatives,positives,per_epi=False,replace_epi=True,replace_TCR =True,neg_multiplier=None,similarity_subset=[["CDR3","Epitope","alpha"]],**kwargs):
    assert len(TCR_pool) >0 
    assert len(Epi_pool) >0
    assert num_negatives >0
    if per_epi:
        assert neg_multiplier

    negatives= pd.DataFrame([])
    if per_epi:
        vc = Epi_pool.value_counts()
        for idx, num_epis in enumerate(vc):
            epitope,mhc =  vc.index[idx]
            
            
            if len(similarity_subset)==1:
                new_neg = set_diff(TCR_pool.drop_duplicates(ignore_index=True),positives[positives.Epitope==epitope][TCR_pool.columns],list(set(TCR_pool.columns)& set(similarity_subset[0])))
            else:
                new_neg = reduce(pd.merge,[set_diff(TCR_pool.drop_duplicates(ignore_index=True),positives[positives.Epitope==epitope][TCR_pool.columns],list(set(TCR_pool.columns)& set(ss))) for ss in similarity_subset])
                
            
            assert len(new_neg)>=neg_multiplier*num_epis ,"possible:"+str(len(new_neg))+" requested: "+ str(neg_multiplier*num_epis) +" Epi: " +epitope
            new_neg = new_neg.sample(neg_multiplier*num_epis,replace=False)
            new_neg["Epitope"] = epitope
            new_neg["MHC A"] = mhc
            negatives = pd.concat([negatives,new_neg])
    else:
        negatives = pd.concat([TCR_pool.sample(n=num_negatives,replace=True).reset_index(drop=True) ,Epi_pool.sample(n=num_negatives,replace=True).reset_index(drop=True)],axis=1).drop_duplicates(ignore_index=True)

        if len(similarity_subset)==1:
            negatives = set_diff(negatives,positives,similarity_subset[0])
        else:
            negatives = reduce(pd.merge,[set_diff(negatives,positives,ss) for ss in similarity_subset])

        rou=0
        
        
        while len(negatives)<num_negatives:
            rou+=1
            if replace_epi and replace_TCR:
                num_new = max(num_negatives-len(negatives),300)
            else:
                num_new = num_negatives-len(negatives)

            new_neg = pd.concat([TCR_pool.sample(n=num_new,replace=True).reset_index(drop=True),Epi_pool.sample(n=num_new,replace=True).reset_index(drop=True)],axis=1).drop_duplicates(ignore_index=True)
            if len(similarity_subset)==1:
                new_neg = set_diff(new_neg,positives,similarity_subset[0])
            else:
                new_neg = reduce(pd.merge,[set_diff(new_neg,positives,ss) for ss in similarity_subset])

            negatives = pd.concat([negatives,new_neg]).drop_duplicates(ignore_index=True)


            if rou>200 and rou%50==0:
                print("not converging warning")
                print('len:' , len(negatives), '/',num_negatives )
      
    
    negatives['Label']=0
    if 'index' in negatives.columns:
        negatives = negatives.drop('index',axis=1)

    return negatives if per_epi else negatives.head(num_negatives) 

     

def shuffle_n_consecutive(arr,n):
    """
    Shuffles the elements of a NumPy array in groups of `n` consecutive elements.

    Parameters:
    arr (NumPy array): The array to shuffle.
    n (int): The number of consecutive elements to shuffle together.

    Returns:
    NumPy array: A new array with the same elements as `arr`, but with the groups of `n` consecutive elements shuffled randomly.
    """
    order = np.array([],dtype=np.int64)
    for i in range(len(arr)//n):
        indexes = np.arange(n) +i*n
        np.random.shuffle(indexes)
        order = np.concatenate([order,indexes])
    indexes = np.arange((len(arr)//n) *n,len(arr)) 
    np.random.shuffle(indexes)
    order =np.concatenate([order,indexes])
    return arr[order]

count=0
def skip_above_count(x,amount,skipped ):
    if x == skipped:
        global count 
        count = count +1
        if count >amount:
            return False
    return True
    
def cap_to_size(df,size=None,output_discarded=False,cap=None):
    """
    caps df to size "size" by discarding datapoints with Epitopes with highest
    frequencies. Datapoints are discarded in a way that preserve frequency order.
    if output_discarded is set to True returns also the discarded datapoints
    """
    assert (size == None and cap) or (cap == None and size)
    
    discarded = pd.DataFrame(columns=df.columns)
    if size and len(df) < size:
        return (df , discarded) if output_discarded else df
    vc = df.Epitope.value_counts()
    
    c = vc.iloc[::-1]
    N=len(c)
    if size:
        for i in range(1,N +1):
            if c[0:i].sum()+c[i-1]*(N-i) > size:
                break
        cap = (size-c[0:i-1].sum())//(N-i+1)
    else:
        i=1
    print(cap)
    
    for cutted_i in range(N-i+1):
        # print(vc.index[cutted_i])
        global count
        count=0
        C_filter = df.Epitope.apply(skip_above_count,amount=cap,skipped=vc.index[cutted_i])
        discarded = discarded.append(df[~ C_filter])
        df = df[C_filter]
    
    return (df , discarded) if output_discarded else df

def required_min_capping(fold,neg_multiplier,TCR_sim=[["CDR3","alpha"]]):
    vc = fold.Epitope.value_counts()
    ret = dict()
    for idx,num_epi in enumerate(vc):
        # n = len(fold[fold.Epitope != vc.index[idx]][["CDR3","alpha"]]) # Wrong does not consider if epitope has same TCR as other epitopes
        # n = len(fold[fold.Epitope != vc.index[idx]][["CDR3","alpha"]].drop_duplicates(ignore_index=True)) #
        if len(TCR_sim)==1:
            n = len(set_diff(fold[TCR_sim[0]].drop_duplicates(ignore_index=True),fold[fold.Epitope==vc.index[idx]][TCR_sim[0]],TCR_sim[0]))
        else:
            
            n = len(reduce(pd.merge,[set_diff(fold.drop_duplicates(ignore_index=True,subset=ss),fold[fold.Epitope==vc.index[idx]],ss) for ss in TCR_sim]))

        print("possible negs for "+ str(vc.index[idx]) ,n)
        if n>=num_epi*neg_multiplier:
            break
        else:
            ret[vc.index[idx]]=n//neg_multiplier
    return ret

def cap_to_PE(fold,neg_multiplier,TCR_sim=[["CDR3","alpha"]]):
    # assert len(fold) == len(set(fold.index)), str(len(fold))+"_"+str( len(set(fold.index)))
    fold = fold.reset_index(drop=True)
    l = len(fold)
    di = required_min_capping(fold,neg_multiplier,TCR_sim=TCR_sim)
    discarded = pd.DataFrame([])
    for k,v in di.items():
        epis = fold[fold.Epitope == k]
        todrop = epis.sample(n=(len(epis)-v))
        fold = fold.drop(todrop.index)
        discarded = pd.concat([discarded,todrop])
    assert l==(len(fold)+len(discarded))
    return fold,discarded




def real_CV(data:pd.DataFrame,task:str,validation =0.125,validation_func="random",folds=5, neg_multiplier=5,
    TCR_tsim = [['CDR3','J','V','alpha','aJ','aV']], # TCR similarity "if two datapoints are the same for TPP tasks"
    TCR_part =['CDR3','J','V','Long','alpha','aJ','aV','aLong'],
    Epi_part = ['Epitope','MHC A'], **kwargs
        )-> List[pd.DataFrame] :
    #negative genareation similarity subset?
    #negative generation of epitope datapoints or disticnt epitopes
    #hardcoded to 5-fold and 5x negative generation 
    
    TCR_sim = kwargs.get("similarity_subset",[["CDR3","alpha"]])
    for ss in TCR_sim:
        if ("Epitope" in ss):
            ss.remove("Epitope") 
        

    """stratified  5-fold cross validation to ensure happiness of Emmi
    data : pd.DataFrame
    task : str ,one of ["TPP1","TPP2","TPP3"]
    validation : float
    splits data to 5 parts where 1 is used as test data for the given task and rest is set to train, if validation is >0.0 train is split into train and validation 
    
    """
    
    
    data['Label'] =1
    train_test_splits=[]
    
    if task =="TPP3" or task == "tpp3":
        epi_counts= data.Epitope.value_counts()
        indexes = np.arange(len(epi_counts))
        indexes = shuffle_n_consecutive(indexes,folds)
        splits_by_epi =[ set(epi_counts.index[indexes[i::folds]]) for i in range(folds)]
        ### Do capping
        splits = [ data[data.Epitope.apply( lambda x: x in splits_by_epi[i])] for i in range(folds)]
        discarded = [ ]
        s = int(len(data)/folds) +1
        for i,split in enumerate(splits):
            if "per_epi" in kwargs and kwargs["per_epi"]:
                splitted ,disc = cap_to_PE(split,neg_multiplier,TCR_sim)
            else:
                splitted ,disc = cap_to_size(split,s,True) 
            splits[i] = splitted
            discarded.append(disc)

        

        for i in range(folds):
            # changed from Set<Tuple> to List<Set<Tuple>>>
            TCRs_in_test= [set([tuple(e[1:]) for e in pd.concat([splits[i],discarded[i]])[ss].itertuples()]) for ss in TCR_tsim]
            # print("TCRs_in_test",len(TCRs_in_test))
            if "use_discarded" in kwargs and kwargs["use_discarded"]:
                train = pd.concat(splits[:i]+splits[i+1:]+discarded[:i]+discarded[i+1:] )
                train = train[train.apply(lambda row: np.all([tuple(row[TCR_tsim[i]]) not in TCRs_in_test[i] for i in range(len(TCR_tsim))]),axis=1)]
                trainn = train.copy()
                if "per_epi" in kwargs and kwargs["per_epi"]:
                    train ,_ = cap_to_PE(train,neg_multiplier,TCR_sim)
                
                
            else:
                train= pd.concat(splits[:i]+splits[i+1:])
                train = train[train.apply(lambda row: np.all([tuple(row[TCR_tsim[i]]) not in TCRs_in_test[i] for i in range(len(TCR_tsim))]),axis=1)]

                trainn = train

            lt=len(train)
            # print("train1",lt)
            
            # print("train2",len(train))
            train_negs = generate_negatives(trainn[TCR_part],train[Epi_part],neg_multiplier*len(train),trainn,neg_multiplier=neg_multiplier,**kwargs)

            test_negs = generate_negatives(pd.concat([splits[i][TCR_part] , discarded[i][TCR_part]]),splits[i][Epi_part],neg_multiplier*len(splits[i]),splits[i],neg_multiplier=neg_multiplier,**kwargs)

            all_train = pd.concat([train_negs,train]).sample(frac=1.0,replace=False)
            all_test = pd.concat([splits[i],test_negs]).sample(frac=1.0, replace=False)
            train_test_splits.append((all_train,all_test))

        


    if task =="TPP2" or task == "tpp2":
        

        TCRs = data[TCR_tsim[0]].drop_duplicates(ignore_index=True).sample(frac=1.0,replace=False)

        splits_by_TCR = [set( tuple(e[1:] for e in TCRs.iloc[i::folds,:].itertuples())) for i in range(folds)]
        
        test_filters = [data.apply(lambda row: tuple(row[TCR_tsim[0]])  in splits_by_TCR[i],axis=1) for i in range(folds) ]

    


        for i in range(folds):
            
            test = data[test_filters[i]]
            train = data[~test_filters[i]]
            TCRs_in_test= [set([tuple(e[1:]) for e in test[ss].itertuples()]) for ss in TCR_tsim[1:]]
            l1 = len(train)
            train = train[train.apply(lambda row: np.all([tuple(row[TCR_tsim[1:][i]]) not in TCRs_in_test[i] for i in range(len(TCR_tsim)-1)]),axis=1)]
            print("TPP2 lost trainpoints: ", l1-len(train))
            # test = datafolds[i]
            # train = pd.concat(datafolds[:i]+datafolds[i+1:])
            train_Epis = set(train.Epitope)
            trainn = train.copy()
            testn = test.copy()
            if "per_epi" in kwargs and kwargs["per_epi"]:
                train ,_ = cap_to_PE(train,neg_multiplier,TCR_sim)
                test,_ = cap_to_PE(test,neg_multiplier,TCR_sim)

            train_negs = generate_negatives(trainn[TCR_part],train[Epi_part],neg_multiplier*len(train),trainn,neg_multiplier=neg_multiplier,**kwargs)

            test_negs = generate_negatives(testn[TCR_part],test[Epi_part],neg_multiplier*len(test),testn,neg_multiplier=neg_multiplier,**kwargs)
        
            all_train = pd.concat([train_negs,train]).sample(frac=1.0,replace=False)
            all_test = pd.concat([test,test_negs]).sample(frac=1.0, replace=False)
            
            
            add_to_train_epi= test[Epi_part][test.Epitope.apply(lambda x:x not in train_Epis)]
            while len(add_to_train_epi) !=0:
                new_neg= generate_negatives(train[TCR_part],add_to_train_epi,len(add_to_train_epi),train,replace_epi=False)
                train_Epis = train_Epis.union(set(new_neg.Epitope))
                all_train = pd.concat([all_train,new_neg])
                
                add_to_train_epi = add_to_train_epi[add_to_train_epi.Epitope.map(lambda x : x not in set(new_neg.Epitope))]
                # print(len(add_to_train_epi))
            train_test_splits.append((all_train,all_test))






    ret =[]
    for train ,test in train_test_splits:
        if validation_func=="random" or task!="tpp3":
            val_filter = np.zeros(len(train),dtype=bool)
            to_val = np.random.choice(len(train), int(len(train)*validation),replace=False)
            val_filter[to_val] = True 
        elif validation_func =="tpp3":
            t_vc = train.Epitope.value_counts()
            to_val=t_vc[(t_vc>=10) & (t_vc<=800)].sample(int(len(t_vc)*validation) ).index
            val_filter = train.Epitope.isin(to_val).to_numpy()

        ret.append((train.iloc[~val_filter,:],train.iloc[val_filter,:],test))

    ret = tuple(zip(*ret))


    for data_list in ret:
        for data in data_list:
            data["Weight"] = epitope_weights(data,**kwargs)*(data["Label"]*(5-1) +1) 
    return ret


def create_save_real_CV(df,dataname,**kwargs):
    if not os.path.exists( os.path.dirname(dataname)):
        print("creating folder: ",os.path.dirname(dataname))
        os.makedirs(os.path.dirname(dataname))
    else:
        print("Adding to existing folder: ", os.path.dirname(dataname))

    if  any([os.path.basename(dataname) in fil for fil in os.listdir(os.path.dirname(dataname)) ]):
        assert False
        inp = input("You are overwriting files named: "+ os.path.basename(dataname) +"*"+ """\n type "yes" to proceed.""")
        
        if inp != "yes":
            print("exiting to avoid overwirting")
            return None
        else:
            print("overwriting files "+ os.path.basename(dataname) +"*")
    assert kwargs["task"] in ["tpp1","tpp2","tpp3","TPP1","TPP2","TPP3","tpp4"]
    train,val,test = real_CV(df,**kwargs)

    dataset_names = ['train','validate']
    if kwargs["task"] == "tpp1" or kwargs["task"] == "TPP1":
        dataset_names.append("test")
    elif kwargs["task"] == "tpp2" or kwargs["task"] == "TPP2":
        dataset_names.append("tpp2")
    elif kwargs["task"] == "tpp3" or kwargs["task"] == "TPP3":
        dataset_names.append("tpp3")
    elif kwargs["task"] == "tpp4" :
        dataset_names.append("test")
    
    for dataset,name in  zip([train,val,test],dataset_names):
        for idx,df in enumerate(dataset):
            if "index" in df.columns:
                df = df.drop("index",axis=1)
            df['MHC class'] =1
            df = df[['CDR3','V','J','Long','alpha','aV','aJ','aLong','MHC A','MHC class','Epitope','Label','Weight']]
            df.to_csv(dataname+str(idx)+ '_' +name+ '_data.gz',index=False,compression='gzip')

def split_to_ERGO(dataname,task):

    for i in range(10):

        dataset = dataname+task+ "_CV/ab_b" + str(i)
        tes = task if task!="tpp1" else "test"
        for t,us in zip(["train","validate",tes],["train","test","real_test"]): 
            datapath ="data/"+dataset+"_"+t+"_data"
            if not os.path.exists(datapath):
                datapath = datapath +".gz"
            if os.path.exists(datapath):
                restricted_samples = pd.read_csv(datapath)
                restricted_samples= restricted_samples[['CDR3','V','J','alpha','aJ','aV','MHC A','MHC class','Epitope','Label']]
                restricted_samples.columns = ["tcrb","vb","jb", "tcra","ja", "va", "mhc","t_cell_type","peptide","sign"]
                restricted_samples["t_cell_type"] = restricted_samples["t_cell_type"].map(lambda x : "MHCI" if x==1 else "MHCII")
                restricted_samples.dropna(subset =["tcrb"],inplace=True)
                

                restricted_samples.fillna("UNK",inplace=True)
                path="../ERGO-II/Samples/"+dataset+"_"+us+"_samples.pickle"
                if not os.path.exists( os.path.dirname(path)):
                    print("creating folder: ",os.path.dirname(path))
                    os.makedirs(os.path.dirname(path))
                filehandler = open(path,"wb")
                pickle.dump(restricted_samples.to_dict('records'),filehandler)
                filehandler.close()
            


def read_and_filter(datasets,require_paired=False,min_epi_freq=None,out_name=None):

    alphabeta = pd.concat([pd.read_csv(dat,dtype = str) for dat in datasets]).drop_duplicates(subset=["Epitope","CDR3","alpha","V","J","aJ","aV","MHC A","Long","aLong"])
   
    
    data_ab = alphabeta[alphabeta['MHC A'].map(is_MHC1)]
    
    print(len(data_ab))
    print("discarding alphas")
    data_ab = data_ab.dropna(subset=["CDR3","V","J"])#,"alpha","aV","aJ"])
    if require_paired:
        data_ab = data_ab.dropna(subset=["alpha","aV","aJ"])
    # data_ab = data_ab[pd.isna(data_ab.CDR3)]
    print(len(data_ab))
    data_ab = data_ab[data_ab.CDR3.map(lambda x: pd.isna(x) or len(x) <=25 )]
    print(len(data_ab))
    data_ab = data_ab[data_ab.alpha.map(lambda x: pd.isna(x) or len(x) <=25 )]
    print(len(data_ab))
    data_ab = data_ab[data_ab.Epitope.map(len)<= 16]
    print(len(data_ab))
    if min_epi_freq:
        vc_e = data_ab.Epitope.value_counts()
        filt = data_ab.Epitope.map( lambda x : vc_e[x]  >= min_epi_freq)
        excluded = data_ab[~filt]
        data_ab = data_ab[filt]
        if out_name:
            ex_file_path = out_name[:-1]+"le"+str(min_epi_freq) + ".gz"
            if not os.path.exists(ex_file_path):
                excluded.to_csv(ex_file_path,index=False,compression='gzip')
    return data_ab



def epitcr_to_titan(file,out_name,epidict,tcrdict):
    data = pd.read_csv(file)
    ret = pd.DataFrame()
    ret["ligand_name"]= data.Epitope.map(lambda x: epidict[x])
    ret["sequence_id"] = data.Long.map(lambda x: tcrdict[x])
    ret["label"] = data.Label
    ret.to_csv(out_name)

def CV_to_titan(name,task):
    epidict_a = dict(pd.read_csv("../TITAN/data/EPItCr_ievdjmcpas_Epis.tsv",header=None,sep="\t").itertuples(index=False))
    tcrdict = dict(pd.read_csv("../TITAN/data/EPItCr_data_beta_long_ievdjmcpas.tsv",header=None,sep="\t").itertuples(index=False))
    for i in range(5):
        folder = name+str(i) + task+"_CV/"
        os.mkdir("../TITAN/"+folder)
        for file in os.listdir(folder):
            base, ext = os.path.splitext(folder+file)
            if ext !=".txt":
                epitcr_to_titan( folder+file,"../TITAN/"+base+".csv",epidict_a,tcrdict)

def aa_to_smiles(seq):
    return Chem.MolToSmiles(Chem.MolFromSequence(seq))

def add_to_titan_data_files(old_longs,old_epis_tsv,newdata,name_long,name_epis_smi,name_epis_tsv):
    data_all = pd.concat(newdata)
    vc_l = data_all.Long.value_counts()
    vc_e = data_all.Epitope.value_counts()
    epidict_t = dict(pd.read_csv(old_epis_tsv,header=None,sep="\t").itertuples(index=False))
    Longdict_t = dict(pd.read_csv(old_longs,header=None,sep="\t").itertuples(index=False))
    vc_l = vc_l[pd.Series(vc_l.index,index=vc_l.index).map(lambda x : x not in Longdict_t)]
    vc_e = vc_e[pd.Series(vc_e.index,index=vc_e.index).map(lambda x : x not in epidict_t)]

    epi_id =vc_e.reset_index().reset_index()
    epi_id.columns = ["id","Epitope","count"]
    epi_id["id"] = epi_id["id"] +max(epidict_t.values())
    epi_id["EpitopeS"] = epi_id["Epitope"].map(aa_to_smiles)
    epi_id[["EpitopeS","id"]].to_csv(name_epis_smi,index=False,sep="\t",header=False)
    epi_id[["Epitope","id"]].to_csv(name_epis_tsv,index=False,sep="\t",header=False)

    tcr_id =vc_l.reset_index().reset_index()
    tcr_id.columns = ["id","beta","count"]
    tcr_id["id"] = tcr_id["id"] + max(Longdict_t.values())

    tcr_id[["beta","id"]].to_csv(name_long,index=False,sep="\t",header=False)

def sublist_list(li):
    aa2=[[]]
    for i in li:
        if i == ',':
            aa2.append([])
        else:
            aa2[-1].append(i)
    return aa2
if __name__ == '__main__':
    
    print("start of main")
    parser = ArgumentParser()
    parser.add_argument("out_name",type=str ,help="Base name of created files")
    parser.add_argument("--dataset","-d",type=str,nargs="+",default=["VDJDB_alphabeta_to_split.csv"],help="list of datasets to use")
    parser.add_argument("--tpp_task",type=int,choices=[1,2],default= 2,help="which task of TPP2 or TPP3")
    parser.add_argument("--validation_func",type=str,default="random",choices=["random","tpp3"],help="how to detrmine validation set 'tpp3' uses unseen epitopes from trianing set")
    parser.add_argument("--folds",type=int, help="number of folds for cross validation")
    parser.add_argument("--neg_multiplier",type=int, help="how many negative datapoints to create per positive datapoint")
    parser.add_argument("--per_epi",action="store_true",help="create negative data per epitope")
    parser.add_argument("--use_discarded",action="store_true",help="use discarded TCRs (in TPP3) for negatve creation (Recommended)")
    parser.add_argument("--require_paired",action="store_true",help="only use data where both alpha and beta chain are present")
    parser.add_argument("--min_epi_freq",type=int,help="minimum number of times an epitope must be seen in the data to be included")
    parser.add_argument("--TCR_tsim",type=str,nargs="+",help="TCR similarity in TPP tasks, creates list of lists where ,separates the lists from each other (earlier just one list)")
    parser.add_argument("--similarity_subset",type=str,nargs="+",help="TCR-Epi datapoint similarity between neg and pos datapoints, creates list of lists where ,separates the lists from each other (earlier just one list)")

    params = parser.parse_args()
    
    params.TCR_tsim = sublist_list(params.TCR_tsim)
    params.similarity_subset = sublist_list(params.similarity_subset)
    print(params)
    data_ab=read_and_filter(params.dataset,params.require_paired,params.min_epi_freq,params.out_name)
    
    # samp=data_ab #.sample(10000)
    
    
    tasks = ["tpp1" ,"tpp2" ,"tpp3","tpp4"]
    if params.tpp_task != -1:
        
        create_save_real_CV(data_ab,params.out_name + tasks[params.tpp_task]+ "_CV/ab_b",task=tasks[params.tpp_task],**vars(params))

    else:
        for task in tasks[:3]:
            create_save_real_CV(data_ab,params.out_name + task+ "_CV/ab_b",task=task)

    with open(params.out_name + tasks[params.tpp_task]+ '_CV/commandline_args.txt', 'w') as f:
        json.dump(params.__dict__, f, indent=2)
    

