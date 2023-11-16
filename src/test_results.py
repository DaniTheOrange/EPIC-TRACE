
import torch
from EPICTRACE_model import LitEPICTRACE, EPICTRACE
import argparse
import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torchmetrics

from split import cap_to_size,read_and_filter
from copy import deepcopy
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score,average_precision_score








def get_path(params,run,idx):

    checkpoint_dir = os.getcwd() + params.folder + params.version[idx] +str(run) +'/checkpoints/'
            
    if not params.SWA:
        index=0
        e=0
        for i,strr in enumerate(os.listdir(checkpoint_dir)):
            start = strr.find("epoch=")
            

            if start != -1:
                ee=int(strr[start+6:strr.find("-")])
                if ee>=e:
                    e=ee
                    index=i
        path = checkpoint_dir+ os.listdir(checkpoint_dir)[index]
    else:
        path = checkpoint_dir +params.version[idx] +str(run)+"_manual_swa"+params.SWA_run +  ".ckpt"
    return path



def manual_testrun(model:torch.nn.Module,dataloader,ret_preds_labels=False):
# def manual_testrun(model:torch.nn.Module,dataloader,device,ret_preds_labels=False):
    # print("manual testing")
    
    

    
    # model.to(device)
    model.eval()
    
    
    pred_all = torch.tensor([],dtype=torch.int)
    # pred_all = torch.tensor([],dtype=torch.int,device=device)
    labels_all = torch.tensor([],dtype=torch.int)
    # labels_all = torch.tensor([],dtype=torch.int,device=device)
    
    for batch in dataloader:
        TCR,Vs,Js,TCR_len ,alpha, aVs,aJs,alpha_len,MHC_As,MHC_class, epitope,epi_len,label,weight = batch
        # TCR,Vs,Js,TCR_len ,alpha, aVs,aJs,alpha_len,MHC_As,MHC_class, epitope,epi_len,label,weight = [inp.to(device) for inp in batch]
        
        # pred =model( TCR,Vs,Js,TCR_len ,alpha, aVs,aJs,alpha_len,MHC_As,MHC_class, epitope,epi_len)
        pred = model(TCR,epitope,alpha,Vs,Js,TCR_len , aVs,aJs,alpha_len,MHC_As,MHC_class, epi_len)

        pred_all = torch.cat([pred_all,pred])
        labels_all = torch.cat([labels_all,deepcopy(label)])
        del TCR,Vs,Js,TCR_len ,alpha, aVs,aJs,alpha_len,MHC_As,MHC_class, epitope,epi_len,label,weight
    try:
        roc_auc = torchmetrics.functional.auroc(pred_all.view(-1),labels_all, pos_label=1).item()
        ap = torchmetrics.functional.average_precision(pred_all.view(-1),labels_all, pos_label=1).item()
    except:
        roc_auc = -1
        ap = -1
    # print("AUROC: " ,roc_auc)
    # print("AP: ", ap)
    if ret_preds_labels:
        return roc_auc,ap,pred_all,labels_all

    return roc_auc,ap



def save_preds(pl_model,model_path,datapath=None,data_embedding_dict=None,savepath=None):
    """saves model predictions to model_path/"""
    
    if datapath is None:
        test_dataloader = pl_model.test_dataloader()
    else:
        if data_embedding_dict is not None:
            with open(data_embedding_dict, 'rb') as handle:
                data_embedding_dict = pickle.load(handle)
        test_dataloader = pl_model.test_dataloader(datapath,data_embedding_dict)
        assert datapath is not None, "datapath must be given if no default test data is given"

#    if type(dataloader) == type([]):
#        dataloader = dataloader[0]
#    if datapath is not None:
#        datase = dataloader.dataset
#        datadf =pd.read_csv(datapath)
#        datase.data = datadf
#        if data_embedding_dict is not None:
#            with open(params.data_embedding_dict, 'rb') as handle:
#                datase.embedding_dict = pickle.load(handle)
#        test_dataloader = DataLoader(datase, batch_size=pl_model.batch_size,
#                        collate_fn=datase.give_collate(pl_model.collate)  ,num_workers=len(os.sched_getaffinity(0)),pin_memory=True)
#    else:
#        test_dataloader=dataloader
    
    roc_auc,ap,pred_all,labels_all =manual_testrun(pl_model.EPICTRACE_model,test_dataloader,True)
    test_data_name = datapath.replace("/","_") if datapath is not None else ""
    if savepath is None:
        savepath= os.path.splitext(model_path)[0]+ test_data_name +"preds.csv"
    print(savepath)
    df = pd.DataFrame({"Labels":labels_all.view(-1),"Predictions":pred_all.view(-1)})
    df.to_csv(savepath,index=False)

def collect_PEpi(path,epitope_list=None,save_version=None,datapath=None):

    #get epitopes from test set and check that results and epis are same length 
    #.../epitcr/loggingDir22/folder22/version_718435/checkpoints/718435_manual_swa_c0.001_1_20_01data_IEVDJcor310fPEpivalR_17_64tpp3_CV_ab_b5_tpp3_data.gzpreds.csv
    #.../epitcr/loggingDir22/folder22/version_718435/checkpoints/718435_manual_swa_c0.001_1_20_01preds.csv
    pv = path.find("version_")
    version=path[pv+8:pv+14]
    task ="TPP" +version[4]
    p1 = path.find("data_")
    if datapath is None:
        if p1 == -1:
            pp1 = path.find("/checkpoints")
            folder = path[:pp1]
            with open(folder+"/hparams.yaml","r") as ff:
                as_string = ff.read()
            pp2 = as_string.find("dataset: ")
            pp3 = as_string.find("\n",pp2+1)
            if datapath is None:
                datapath = "data/"+ as_string[pp2+9:pp3] + "_" + task.lower()  +"_data.gz"
            
        else:
            p2 = path.find("preds.csv")
            if datapath is None:
                datapath = "data/" + path[p1+5:p2].replace("data_","data/").replace("CV_ab","CV/ab").replace("final_alpha_ievdj","final_alpha/ievdj")
    if save_version is None:
        save_version = os.path.split(path)[1].replace("manual_swa_c","").replace("manual_swa_","").replace("_01preds.csv","")
    test_df = pd.read_csv(datapath)
    
    scores_df = pd.read_csv(path)

    
        
    if epitope_list is None:
        epitope_list = test_df.Epitope.drop_duplicates()
    
    ret ={task+"_ROC_AUC":{},task+"_ap": {}}
    print("epi list len",len(epitope_list))

    for epitope in epitope_list:

        filt= test_df.Epitope ==epitope
        try:
            auroc = roc_auc_score(scores_df[filt].Labels,scores_df[filt].Predictions)
            ap = average_precision_score(scores_df[filt].Labels,scores_df[filt].Predictions)
            # auroc ,ap = manual_testrun(pl_model.epiTCR,dataloader,device)
            print(epitope, auroc, ap)
            ret[task+"_ROC_AUC"][epitope] =auroc
            ret[task+"_ap"][epitope]=ap
        except Exception as ex:
            print(epitope,ex)
        
        
    # print(ret)
    sd = sorted(list(ret[task+"_ap"].items()), key= lambda x: x[1])
    print(sd[:5])
    print(sd[-6:])
    keys,vals = zip(*sd)
    print(np.mean(vals))

    sdr = sorted(list(ret[task+"_ROC_AUC"].items()), key= lambda x: x[1])
    
    keysr,valsr = zip(*sdr)
    # ret[tasks[idx]+"_ROC_AUC"] = pd.Series(valsr,index=keysr)
    # ret[tasks[idx]+"_ap"] = pd.Series(vals,index=keys)
    
        
        
    r_s = pd.Series(valsr,index=keysr,name=save_version)
    a_s = pd.Series(vals,index=keys,name=save_version)
    c_s = pd.Series(test_df.Epitope.value_counts(),name=save_version)

    
    
    

    for s,nam in zip([r_s,a_s,c_s],["auroc","ap","counts"]):
        try:
            savename = os.getcwd()+"/results/PEpi_result_series/"+ save_version +"_" +nam+"_"+task +".bin"
            

            s.to_pickle(savename)
        except:
            print("something went wrong in pickling")
        
   
def collect_epitopes(path, epitope_list, outname, save_version=None,datapath=None):
    pv = path.find("version_")
    version = path[pv+8:pv+14]
    task = "TPP" + version[4]
    p1 = path.find("data_")
    if datapath is None:
        if p1 == -1:
            pp1 = path.find("/checkpoints")
            folder = path[:pp1]
            with open(folder+"/hparams.yaml","r") as ff:
                as_string = ff.read()
            pp2 = as_string.find("dataset: ")
            pp3 = as_string.find("\n",pp2+1)
            if datapath is None:
                datapath = "data/"+ as_string[pp2+9:pp3] + "_" + task.lower()  +"_data.gz"
            
        else:
            p2 = path.find("preds.csv")
            if datapath is None:
                datapath = "data/" + path[p1+5:p2].replace("data_","data/").replace("CV_ab","CV/ab")
    if save_version is None:
        save_version = os.path.split(path)[1].replace("manual_swa_c","").replace("manual_swa_","").replace("_01preds.csv","") + outname
    test_df = pd.read_csv(datapath)
    
    scores_df = pd.read_csv(path)
    if epitope_list is None:
        filt = np.ones(len(scores_df),dtype=bool)
    else:
        episet = set(epitope_list)
        filt = test_df.Epitope.map(lambda x : x in episet)

    auroc = roc_auc_score(scores_df[filt].Labels,scores_df[filt].Predictions)
    ap = average_precision_score(scores_df[filt].Labels,scores_df[filt].Predictions)
    rdict = dict()
    rdict[task+"_ROC_AUC"] = auroc
    rdict[task+"_ap"] = ap
    print(rdict)
    with open("results/result_dicts/"+ save_version + ".bin",'wb') as handle:
        pickle.dump(rdict,handle)

if __name__ =='__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('version', type=str ,nargs="+")
    parser.add_argument('--folder', type=str,default='/loggingDir22/folder22/version_')
    parser.add_argument('--runs', type=int,nargs="+",default=[0,1,2,3,4])
    parser.add_argument("--SWA",action= "store_true",default = False)
    parser.add_argument("--SWA_run",type=str)
    parser.add_argument("--out_name",default="outResult",help="name added to the output file")
    
    parser.add_argument("--dataset",type=str,help="path to dataset")
    parser.add_argument("--data_embedding_dict",type=str)
    parser.add_argument("--save_preds",action="store_true")
    parser.add_argument("--collect_PEpi_path",type=str)
    parser.add_argument("--collect_path",type=str)
    parser.add_argument("--gepi15",action="store_true")
    parser.add_argument("--save_version",type=str,help="Version name added to the save file")
    parser.add_argument("--pred_save_path",type=str,help="path to save the predictions")

    params = parser.parse_args()

    assert (params.SWA and params.SWA_run) or ( not params.SWA)
    if params.collect_PEpi_path is not None:
        collect_PEpi(params.collect_PEpi_path,save_version=params.save_version,datapath=params.dataset)
    elif params.collect_path is not None:
        if params.gepi15:
            ab_data =read_and_filter(["data/VDJDB_to_split_2022-06-17_corrected.csv","data/IEDB_to_split_2022-06-17_corrected.csv"]).drop(["MHC class","MHC B"],axis=1)
            e_vc = ab_data.Epitope.value_counts()
            gepis = set(e_vc[e_vc>=15].index)
        else:
            gepis = None
        collect_epitopes(params.collect_path,gepis,params.out_name,params.save_version,params.dataset)
    else:
        print("pl version ",pl.__version__)
        print(torch.cuda.device_count())
        trainer = pl.Trainer(gpus =torch.cuda.device_count(),reload_dataloaders_every_epoch=True)

        for version_idx,version in enumerate(params.version):

            for idx,run in enumerate(params.runs): 

                path = get_path(params,run,version_idx)
                print(path)
                model = LitEPICTRACE.load_from_checkpoint(path,"cpu")
                
                model.eval()
                model.freeze()
                if params.save_preds:
                    save_preds(model,path,params.dataset,params.data_embedding_dict,params.pred_save_path)
                        
