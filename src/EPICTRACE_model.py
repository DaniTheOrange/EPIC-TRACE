
import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score ,average_precision_score
from torch.nn.modules.activation import MultiheadAttention
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import subprocess
import torchmetrics
import os
import copy
from construct_long import correct_tolong

second_level = ['HLA-B*44', 'HLA-A*03', 'HLA-B*37', 'HLA-B*35', 'HLA-A*11', 'HLA-B*40', 'HLA-A*30', 'HLA-B*52', 'HLA-B*51', 'HLA-C*14', 'HLA-B*81', 'HLA-B*57', 'HLA-B*58', 'HLA-C*03', 'HLA-C*07', 'HLA-B*07', 'HLA-C*08', 'HLA-C*04', 'HLA-A*32', 'HLA-A*02', 'HLA-B*18',
 'HLA-A*2', 'HLA-A*25', 'HLA-A*24', 'HLA-B*14', 'HLA-B*42', 'HLA-B*12', 'HLA-B*7', 'HLA-B*08', 'HLA-B*15', 'HLA-E*01', 'HLA-B*38', 'HLA-B*53', 'HLA-A*1', 'HLA-C*05', 'HLA-C*06', 'HLA-B*8', 'HLA-A*68', 'HLA-C*w3', 'HLA-A*01', 'HLA-B*27','HLA-A*29']
third_level = ['HLA-B*51:01', 'HLA-A*01:01', 'HLA-B*14:02', 'HLA-B*42:01', 'HLA-B*35:01', 'HLA-C*04:01', 'HLA-B*37:01', 'HLA-C*03:04', 'HLA-B*38:01', 'HLA-A*24:02', 'HLA-B*81:01', 'HLA-C*07:01', 'HLA-A*02:256', 'HLA-A*03:01', 'HLA-B*35:42',
  'HLA-B*27:05', 'HLA-C*07:02', 'HLA-A*25:01', 'HLA-B*51:193', 'HLA-C*14:02', 'HLA-B*57:01', 'HLA-B*52:01', 'HLA-A*11:01', 'HLA-B*18:01', 'HLA-B*57:06', 'HLA-A*32:01', 'HLA-C*03:03', 'HLA-B*08:01',
  'HLA-B*07:02', 'HLA-C*08:02', 'HLA-E*01:03', 'HLA-A*02:01', 'HLA-B*15:01', 'HLA-B*44:03', 'HLA-C*06:02', 'HLA-B*40:01', 'HLA-B*44:02', 'HLA-B*35:08', 'HLA-B*57:03', 'HLA-E*01:01', 'HLA-C*05:01', 'HLA-A*68:01', 'HLA-A*30:02', 'HLA-B*44:05',
  'HLA-A*02:06', 'HLA-A*02:266', 'HLA-A*24:01', 'HLA-A*29:02']

def add_asterisk(hla1):
    if len(hla1) >5:
        if hla1[5] != '*':
            return hla1[:5] + '*' + hla1[5:]
    return hla1


class PPIDataset2(Dataset):
    
    def __init__(self,csv_file=None,df=None,embedding_dict=None,output_mhc_A=False,output_vj=False,mhc_hi=False,add_01=False,only_CDR3=False,load_MHC_dicts=False,mhc_dict_path="data/MHC_all_dict.bin"):
        assert(csv_file is not None or df is not None)
        
        if csv_file:
            
            try:
                self.data = pd.read_csv(csv_file)
            except:
                self.data = pd.read_csv(csv_file+".gz")

            print("dataloaded happily")
        else:
            self.data = df.copy()
        if add_01:
            cols=self.data.columns
            
            newdata = self.data.apply(correct_tolong,axis=1)
            print("added longs: ", newdata.apply(lambda row: len(row["Long"]) > len(row["CDR3"]),axis=1).sum() - self.data.apply(lambda row: len(row["Long"]) > len(row["CDR3"]),axis=1).sum())
            self.data = newdata[cols]
            
        

        if type(embedding_dict) == dict:
            self.embedding_dict = embedding_dict
        elif type(embedding_dict) == str:
            with open(embedding_dict, 'rb') as handle:
                self.embedding_dict = pickle.load(handle)
        
        self.output_mhc_A = output_mhc_A
        self.mhc_hi = mhc_hi and output_mhc_A
        self.output_vj = output_vj
        self.only_CDR3 = only_CDR3
        self.aa_dict=dict(zip(list("GPAVLIMCFYWHKRQNEDST"),list(range(1,21))))
        
        if load_MHC_dicts:
            with open(mhc_dict_path,"rb") as handle:
                self.mhc_dict = pickle.load(handle)
        else:
            self.mhc_dict =dict(zip([None,'HLA class I', 'HLA class II', 'HLA-A*01', 'HLA-A*01:01', 'HLA-A*02', 'HLA-A*02:01', 'HLA-A*02:01:110', 'HLA-A*02:01:48', 'HLA-A*02:01:59', 'HLA-A*02:01:98',
            'HLA-A*02:256', 'HLA-A*03', 'HLA-A*03:01', 'HLA-A*11', 'HLA-A*11:01', 'HLA-A*24:02', 'HLA-A*24:02:84', 'HLA-A*25:01', 'HLA-A*30:02', 'HLA-A*32:01', 'HLA-A*68:01', 'HLA-A1', 'HLA-A11',
            'HLA-A2', 'HLA-B*07', 'HLA-B*07:02', 'HLA-B*08', 'HLA-B*08:01', 'HLA-B*08:01:29', 'HLA-B*12', 'HLA-B*14:02', 'HLA-B*15', 'HLA-B*15:01', 'HLA-B*18', 'HLA-B*18:01', 'HLA-B*27', 'HLA-B*27:05',
            'HLA-B*27:05:31', 'HLA-B*35', 'HLA-B*35:01', 'HLA-B*35:08', 'HLA-B*35:08:01', 'HLA-B*35:42:01', 'HLA-B*37:01', 'HLA-B*38:01', 'HLA-B*40:01', 'HLA-B*42', 'HLA-B*42:01', 'HLA-B*44',
                'HLA-B*44:02', 'HLA-B*44:03:08', 'HLA-B*44:05', 'HLA-B*44:05:01', 'HLA-B*51:01', 'HLA-B*51:193', 'HLA-B*52:01', 'HLA-B*53', 'HLA-B*57', 'HLA-B*57:01', 'HLA-B*57:03','HLA-B*57:06',
                'HLA-B*58', 'HLA-B*81:01', 'HLA-B18', 'HLA-B27', 'HLA-B35', 'HLA-B57', 'HLA-B7', 'HLA-B8', 'HLA-C*03:03', 'HLA-C*03:04', 'HLA-C*04:01', 'HLA-C*05:01', 'HLA-C*06:02', 'HLA-C*07:01',
                'HLA-C*07:02', 'HLA-C*08:02', 'HLA-C*14:02', 'HLA-Cw3', 'HLA-DPA1*01:03', 'HLA-DPA1*02:01', 'HLA-DPB1*02:01', 'HLA-DQ', 'HLA-DQ1', 'HLA-DQA1*01:02', 'HLA-DQA1*01:03', 'HLA-DQA1*03:01',
                'HLA-DQA1*03:01:01', 'HLA-DQA1*05:01', 'HLA-DQA1*05:01:01:02', 'HLA-DQB1*03:02', 'HLA-DR', 'HLA-DR1', 'HLA-DR3', 'HLA-DRA*01', 'HLA-DRA*01:01', 'HLA-DRA*01:01:02', 'HLA-DRA*01:02:03',
                    'HLA-DRA1*01', 'HLA-DRB1', 'HLA-DRB1*01:01', 'HLA-DRB1*03:01', 'HLA-DRB1*04:01', 'HLA-DRB1*04:04', 'HLA-DRB1*04:05', 'HLA-DRB1*07:01', 'HLA-DRB1*11:01', 'HLA-DRB1*14:02', 'HLA-DRB1*15:01',
                    'HLA-DRB1*15:03', 'HLA-DRB3*01:01', 'HLA-DRB3*03:01', 'HLA-DRB4*01:01', 'HLA-E*01:01', 'HLA-E*01:01:01:03', 'HLA-E*01:03','HLA-C*08:02:12', 'HLA-A*24:01', 'HLA-A*29:02', 'HLA-B*44:03', 'HLA-A*02:266',
                    'HLA-A*11:01:18', 'HLA-B*07:02:48', 'HLA-A*01:01:73', 'HLA-B*35:42:02', 'HLA-B*35:01:45', 'HLA-B*37:01:10', 'HLA-A*02:06:01:03', 'HLA-A*24:02:33'],
                    list(range(0,117))))
        if load_MHC_dicts:
            with open("data/MHC_lvl2nd_dict.bin","rb") as handle:
                self.dict2 = pickle.load(handle)
                # print(len(self.dict2),type(self.dict2))
            with open("data/MHC_lvl3rd_dict.bin","rb") as handle:
                self.dict3 = pickle.load(handle)
        else:        
            self.dict2 = dict(zip(second_level,list(range(1,len(second_level)+1))))
            self.dict3 = dict(zip(third_level,list(range(1,len(third_level)+1))))

        def get_genefamily(gene):
            pos = gene.find("-")
            if pos==-1:
                return gene
            else:
                return gene[:pos]
        vj_dict = dict()

        for gene in ["aJ","aV","bJ","bV"]:
            
            fi = pd.read_csv('data/tr' + gene.lower() +'s.tsv',delimiter='\t').Gene.map(get_genefamily).drop_duplicates()
            sec = pd.read_csv('data/tr' + gene.lower() +'s.tsv',delimiter='\t').Gene.drop_duplicates()
            trd = pd.read_csv('data/tr' + gene.lower() +'s.tsv',delimiter='\t').Allele.drop_duplicates()
            vj_dict[gene if not "b" in gene else gene[-1]] = pd.concat([fi,sec,trd]).drop_duplicates()
        self.v_dict =  dict(zip(vj_dict["V"],list(range(1,166))))
        self.j_dict =  dict(zip(vj_dict["J"],list(range(1,30))))
        self.av_dict =  dict(zip(vj_dict["aV"],list(range(1,162))))
        self.aj_dict =  dict(zip(vj_dict["aJ"],list(range(1,112))))

        self.v_dict[None] =0 
        self.j_dict[None] =0 
        self.av_dict[None] =0 
        self.aj_dict[None] =0 

    
        self.TCR_max_length = self.data.CDR3.map(lambda x :len(x) if pd.notna(x) else 0).max()

        self.epitope_max_length = self.data.Epitope.map(len).max()
        
    def get_data(self):
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
       ##### NOTE aJ and aV wrong way as in cv
        
        CDR3,V,J,long,alpha,aV,aJ,along,MHC_A,MHC_class,epitope, label, weight= tuple(self.data.iloc[idx,:])
        return CDR3,V,J,long,alpha,aV,aJ,along,MHC_A,MHC_class,epitope,label,weight
    
    def get_hla_pattern(self,hla):
        class_1 = ['HLA-A' , 'HLA-B' , 'HLA-C' , 'HLA-E' ]
        class_2 = ['HLA class II'] # not working yet
        if hla == 'HLA class I':
            return (1,0,0,0)
        ret=[0,0,0,0]
        for cl1 in class_1:
            if cl1 in hla:
                ret[0]=1
        hla = add_asterisk(hla)
        dict1 = dict(zip(class_1,list(range(1,5))))
        

        i = hla.find('*')
        hla_l1 = hla[:i] if i!=-1 else hla
        if  hla_l1 in dict1:
            ret[1]=dict1[hla_l1]

        i = hla.find(':')
        hla_l2 = hla[:i] if i!=-1 else hla
        if  hla_l2 in self.dict2:
            ret[2]=self.dict2[hla_l2]

        
        i = hla.find(':')
        if i >-1:
            i = hla.find(':',i+1)
            
        
            hla_l3 = hla[:i] if i!=-1 else hla

            if  hla_l3 in self.dict3:
                ret[3]=self.dict3[hla_l3]

        return tuple(ret)

    def one_hot_collate_fn(self,batch):
        
        # betas,CDR3s,Vs,Js,MHC_As,MHC_class,epitopes,longs, labels = list(map(list, zip(*batch)))
        CDR3s_in,Vs,Js,longs, alphas_in,aVs,aJs,ALongs,MHC_As,MHC_class,epitopes, labels,weight = list(map(list, zip(*batch)))
#         CDR3s = list(CDR3s)
        CDR3s = [torch.tensor([self.aa_dict[aa] for aa in CDR3],dtype=torch.int64) if pd.notna(CDR3) else torch.zeros(self.TCR_max_length,dtype=torch.int64) for CDR3 in CDR3s_in]
        CDR3s[0] = torch.cat([CDR3s[0],torch.zeros(self.TCR_max_length-len(CDR3s[0]),dtype=torch.int64)])
        padded_CDR3s =torch.nn.utils.rnn.pad_sequence(CDR3s,True,0)
        one_hot_padded_CDR3s = torch.nn.functional.one_hot(padded_CDR3s,21).permute(0,2,1).to(torch.float)

        alphas = [torch.tensor([self.aa_dict[aa] for aa in alpha],dtype=torch.int64) if pd.notna(alpha) else torch.zeros(self.TCR_max_length,dtype=torch.int64) for alpha in alphas_in]
        alphas[0] = torch.cat([alphas[0],torch.zeros(self.TCR_max_length-len(alphas[0]),dtype=torch.int64)])
        padded_alphas =torch.nn.utils.rnn.pad_sequence(alphas,True,0)
        one_hot_padded_alphas = torch.nn.functional.one_hot(padded_alphas,21).permute(0,2,1).to(torch.float)

        
        epitopes = [ torch.tensor([self.aa_dict[aa] for aa in epitope],dtype=torch.int64) for epitope in epitopes]
        epitopes[0] = torch.cat([epitopes[0],torch.zeros(self.epitope_max_length-len(epitopes[0]),dtype=torch.int64)])
        padded_epitopes = torch.nn.utils.rnn.pad_sequence(epitopes,True,0)
        one_hot_padded_epitopes = torch.nn.functional.one_hot(padded_epitopes,21).permute(0,2,1).to(torch.float)
        ret = [one_hot_padded_CDR3s,None,None,None ,one_hot_padded_alphas,None,None ,None ,None,None, one_hot_padded_epitopes,None,None,None]
        
        cdr3_lens = torch.tensor([len(e) if pd.notna(e) else 0 for e in CDR3s_in])
        alpha_lens = torch.tensor([len(e) if pd.notna(e) else 0 for e in alphas_in])
        epi_lens = torch.tensor([len(e) for e in epitopes])
        ret[3] = cdr3_lens
        ret[7] = alpha_lens
        ret[11] = epi_lens
        

        if self.output_vj:
            ret[1] = torch.tensor([self.v_dict[v] if v in self.v_dict else 0 for v in Vs],dtype=torch.int64).view(-1,1)
            ret[2] = torch.tensor([self.j_dict[j] if j in self.j_dict else 0 for j in Js],dtype=torch.int64).view(-1,1)
            ret[5] = torch.tensor([self.av_dict[v] if v in self.av_dict else 0 for v in aVs],dtype=torch.int64).view(-1,1)
            ret[6] = torch.tensor([self.aj_dict[j] if j in self.aj_dict else 0 for j in aJs],dtype=torch.int64).view(-1,1)
        

        
        if self.output_mhc_A:
            if self.mhc_hi:
                ret[8] = torch.cat([torch.tensor(level,dtype=torch.int64).view(-1,1) for level in zip(*[self.get_hla_pattern(mhc)  for mhc in MHC_As]) ],dim=1)
            else:
                ret[8] =torch.tensor([self.mhc_dict[mhc] if mhc in self.mhc_dict else 0  for mhc in MHC_As],dtype=torch.int64).view(-1,1)
        
            
        ret[12] =torch.tensor(labels)
        ret[13] =torch.tensor(weight)
        # ret == [one_hot_padded_CDR3s,Vs,Js,cdr3_len ,one_hot_padded_alphas, aVs,aJs,alpha_len,MHC_As,MHC_class, one_hot_padded_epitopes,epi_len,label]
        return ret
    
    def embedding_collate_fn(self,batch):
        # betas,CDR3s,Vs,Js,MHC_As,MHC_class,epitopes,longs, labels = list(map(list, zip(*batch)))
        # CDR3s,Vs,Js,MHC_As,MHC_class,epitopes,longs, labels = list(map(list, zip(*batch)))
        CDR3s_in,Vs,Js,longs, alphas_in,aVs,aJs,ALongs,MHC_As,MHC_class,epitopes, labels,weight = list(map(list, zip(*batch)))
        
        if self.only_CDR3:
            longs = CDR3s_in
            ALongs = alphas_in
            
#         longs_tensors = [torch.unsqueeze(torch.tensor(self.embedding_dict[e]).permute(1,0),0) for e in longs]
        sheep_tensor = torch.tensor(next(iter(self.embedding_dict.items()))[1]) 
        longs_tensors = [torch.tensor(self.embedding_dict[e]) if pd.notna(e) else torch.zeros_like(sheep_tensor )for e in longs]
        
        
        longs_tensors[0] = torch.cat([longs_tensors[0],torch.zeros((self.TCR_max_length-len(longs_tensors[0]),longs_tensors[0].shape[1]))],dim=0)
        padded_longs = torch.nn.utils.rnn.pad_sequence(longs_tensors,True,0).permute(0,2,1).to(torch.float)

        alongs_tensors = [torch.tensor(self.embedding_dict[e]) if pd.notna(e) else torch.zeros_like( sheep_tensor )for e in ALongs]        
        alongs_tensors[0] = torch.cat([alongs_tensors[0],torch.zeros((self.TCR_max_length-len(alongs_tensors[0]),alongs_tensors[0].shape[1]))],dim=0)
        padded_alongs = torch.nn.utils.rnn.pad_sequence(alongs_tensors,True,0).permute(0,2,1).to(torch.float)
        
#         epitopes_tensor = torch.cat([ torch.unsqueeze(torch.tensor(self.embedding_dict[e]).permute(1,0),0) for e in epitopes])
        epitopes_tensors = [ torch.tensor(self.embedding_dict[e]) if pd.notna(e) else torch.zeros( (2,sheep_tensor.shape[1]) ) for e in epitopes]
        
        epitopes_tensors[0] = torch.cat([epitopes_tensors[0],torch.zeros((self.epitope_max_length-len(epitopes_tensors[0]),epitopes_tensors[0].shape[1]))],dim=0)
        padded_epitopes = torch.nn.utils.rnn.pad_sequence(epitopes_tensors,True,0).permute(0,2,1).to(torch.float)
        
        ret = [padded_longs,None,None,None ,padded_alongs,None,None ,None ,None,None, padded_epitopes,None,None,None]

        
        cdr3_lens = torch.tensor([len(e) if pd.notna(e) else 0 for e in CDR3s_in])
        alpha_lens = torch.tensor([len(e) if pd.notna(e) else 0 for e in alphas_in])
        epi_lens = torch.tensor([len(e) for e in epitopes])
        ret[3]=cdr3_lens
        ret[7] = alpha_lens
        ret[11] = epi_lens
        

        if self.output_vj:
            ret[1] = torch.tensor([self.v_dict[v] if v in self.v_dict else 0 for v in Vs],dtype=torch.int64).view(-1,1)
            ret[2] = torch.tensor([self.j_dict[j] if j in self.j_dict else 0 for j in Js],dtype=torch.int64).view(-1,1)
            ret[5] = torch.tensor([self.av_dict[v] if v in self.av_dict else 0 for v in aVs],dtype=torch.int64).view(-1,1)
            ret[6] = torch.tensor([self.aj_dict[j] if j in self.aj_dict else 0 for j in aJs],dtype=torch.int64).view(-1,1)
        

        if self.output_mhc_A:
            
            if self.mhc_hi:
                ret[8] = torch.cat([torch.tensor(level,dtype=torch.int64).view(-1,1) for level in zip(*[self.get_hla_pattern(mhc)  for mhc in MHC_As]) ],dim=1)
            else:
                ret[8] =torch.tensor([self.mhc_dict[mhc] if mhc in self.mhc_dict else 0  for mhc in MHC_As],dtype=torch.int64).view(-1,1)
                
        
            
        ret[12] =torch.tensor(labels)
        ret[13] =torch.tensor(weight)
        # ret == [one_hot_padded_CDR3s,Vs,Js,cdr3_len ,one_hot_padded_alphas, aVs,aJs,alpha_len,MHC_As,MHC_class, one_hot_padded_epitopes,epi_len,label]
        return ret



    def embedding_one_hot_collate(self,batch):

        CDR3s_in,Vs,Js,longs, alphas_in,aVs,aJs,ALongs,MHC_As,MHC_class,epitopes, labels,weight = list(map(list, zip(*batch)))
        
        if self.only_CDR3:
            longs = CDR3s_in
            ALongs = alphas_in

        sheep_tensor = torch.as_tensor(next(iter(self.embedding_dict.items()))[1],dtype=torch.float) 
        longs_tensors = [torch.as_tensor(self.embedding_dict[e]) if pd.notna(e) else torch.zeros_like(sheep_tensor )for e in longs]
        
        
        longs_tensors[0] = torch.cat([longs_tensors[0],torch.zeros((self.TCR_max_length-len(longs_tensors[0]),longs_tensors[0].shape[1]))],dim=0)
        padded_longs = torch.nn.utils.rnn.pad_sequence(longs_tensors,True,0).permute(0,2,1).to(torch.float)

        alongs_tensors = [torch.as_tensor(self.embedding_dict[e]) if pd.notna(e) else torch.zeros_like( sheep_tensor )for e in ALongs]        
        alongs_tensors[0] = torch.cat([alongs_tensors[0],torch.zeros((self.TCR_max_length-len(alongs_tensors[0]),alongs_tensors[0].shape[1]))],dim=0)
        padded_alongs = torch.nn.utils.rnn.pad_sequence(alongs_tensors,True,0).permute(0,2,1).to(torch.float)
        
        epitopes_tensors = [ torch.as_tensor(self.embedding_dict[e]) if pd.notna(e) else torch.zeros( (2,sheep_tensor.shape[1]) ) for e in epitopes]
        epitopes_tensors[0] = torch.cat([epitopes_tensors[0],torch.zeros((self.epitope_max_length-len(epitopes_tensors[0]),epitopes_tensors[0].shape[1]))],dim=0)
        padded_epitopes = torch.nn.utils.rnn.pad_sequence(epitopes_tensors,True,0).permute(0,2,1).to(torch.float)

        CDR3s = [torch.tensor([self.aa_dict[aa] for aa in CDR3],dtype=torch.int64) if pd.notna(CDR3) else torch.zeros(self.TCR_max_length,dtype=torch.int64) for CDR3 in CDR3s_in]
        CDR3s[0] = torch.cat([CDR3s[0],torch.zeros(self.TCR_max_length-len(CDR3s[0]),dtype=torch.int64)])
        padded_CDR3s =torch.nn.utils.rnn.pad_sequence(CDR3s,True,0)
        one_hot_padded_CDR3s = torch.nn.functional.one_hot(padded_CDR3s,21).permute(0,2,1).to(torch.float)

        alphas = [torch.tensor([self.aa_dict[aa] for aa in alpha],dtype=torch.int64) if pd.notna(alpha) else torch.zeros(self.TCR_max_length,dtype=torch.int64) for alpha in alphas_in]
        alphas[0] = torch.cat([alphas[0],torch.zeros(self.TCR_max_length-len(alphas[0]),dtype=torch.int64)])
        padded_alphas =torch.nn.utils.rnn.pad_sequence(alphas,True,0)
        one_hot_padded_alphas = torch.nn.functional.one_hot(padded_alphas,21).permute(0,2,1).to(torch.float)

        epitopes = [ torch.tensor([self.aa_dict[aa] for aa in epitope],dtype=torch.int64) for epitope in epitopes]
        epitopes[0] = torch.cat([epitopes[0],torch.zeros(self.epitope_max_length-len(epitopes[0]),dtype=torch.int64)])
        padded_epitopes_o = torch.nn.utils.rnn.pad_sequence(epitopes,True,0)
        one_hot_padded_epitopes = torch.nn.functional.one_hot(padded_epitopes_o,21).permute(0,2,1).to(torch.float)
        
        ret = [torch.cat([padded_longs,one_hot_padded_CDR3s],dim=1),None,None,None ,torch.cat([padded_alongs,one_hot_padded_alphas],dim=1),None,None ,None ,None,None, torch.cat([padded_epitopes,one_hot_padded_epitopes],dim=1),None,None,None]
        
        cdr3_lens = torch.tensor([len(e) if pd.notna(e) else 0 for e in CDR3s_in])
        alpha_lens = torch.tensor([len(e) if pd.notna(e) else 0 for e in alphas_in])
        epi_lens = torch.tensor([len(e) for e in epitopes])
        ret[3]=cdr3_lens
        ret[7] = alpha_lens
        ret[11] = epi_lens
        

        if self.output_vj:
            ret[1] = torch.tensor([self.v_dict[v] if v in self.v_dict else 0 for v in Vs],dtype=torch.int64).view(-1,1)
            ret[2] = torch.tensor([self.j_dict[j] if j in self.j_dict else 0 for j in Js],dtype=torch.int64).view(-1,1)
            ret[5] = torch.tensor([self.av_dict[v] if v in self.av_dict else 0 for v in aVs],dtype=torch.int64).view(-1,1)
            ret[6] = torch.tensor([self.aj_dict[j] if j in self.aj_dict else 0 for j in aJs],dtype=torch.int64).view(-1,1)
        

        if self.output_mhc_A:
            if self.mhc_hi:
                ret[8] = torch.cat([torch.tensor(level,dtype=torch.int64).view(-1,1) for level in zip(*[self.get_hla_pattern(mhc)  for mhc in MHC_As]) ],dim=1)
            else:
                ret[8] =torch.tensor([self.mhc_dict[mhc] if mhc in self.mhc_dict else 0  for mhc in MHC_As],dtype=torch.int64).view(-1,1)


        
            
        ret[12] =torch.tensor(labels)
        ret[13] =torch.tensor(weight)

        return ret
        
    def give_collate(self,collate_name):
        if collate_name == "embedding":
            return self.embedding_collate_fn

        elif collate_name == "one_hot":
            return self.one_hot_collate_fn
        elif collate_name == "embedding_one_hot":
            return self.embedding_one_hot_collate
        elif collate_name == "index":
            return self.index_collate_fn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads,output_attn_w=False, n_hidden=64, dropout=0.1):
        """
        Args:
          embed_dim: Number of input and output features.
          n_heads: Number of attention heads in the Multi-Head Attention.
          n_hidden: Number of hidden units in the Feedforward (MLP) block.
          dropout: Dropout rate after the first layer of the MLP and in two places on the main path (before
                   combining the main path with a skip connection).
        """
        super(TransformerBlock,self).__init__()
        self.mh = nn.MultiheadAttention(embed_dim,n_heads,dropout=dropout)
        self.drop1= nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop2= nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim,n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hidden,embed_dim))
        self.output_attn_w = output_attn_w

    def forward(self, x):
        """
        Args:
          x of shape (max_seq_length, batch_size, embed_dim): Input sequences.
          
        
        Returns:
          xm of shape (max_seq_length, batch_size, embed_dim): Encoded input sequences.
          attn_w of shape (batch_size, max_seq_length, max_seq_length)
        """
        xm, attn_w= self.mh(x,x,x)
        xm = self.drop1(xm)
        xm = self.norm1(x+xm)
        x = self.ff(xm)
        x = self.drop2(x)
        xm = self.norm2(x+xm)
        return (xm,attn_w) if self.output_attn_w else xm

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads,dropout=0):
        super(SelfAttention,self).__init__()
        self.mh = nn.MultiheadAttention(embed_dim,n_heads,dropout=dropout)

    def forward(self,x):
        """
        Args:
        x of shape (max_seq_length, batch_size, embed_dim): Input sequences.
        
        
        Returns:
        xm of shape (max_seq_length, batch_size, embed_dim): Encoded input sequences.
        attn_w of shape (batch_size, max_seq_length, max_seq_length)
        """
        xm, attn_w= self.mh(x,x,x)
        return xm,attn_w

class EPICTRACE(nn.Module):
    def __init__(self,params={}):
        
        
        super(EPICTRACE, self).__init__()
        
        
        
        self.TCR_max_length = params["TCR_max_length"] if "TCR_max_length" in params else 25
        self.epitope_max_length = params["epitope_max_length"] if "epitope_max_length" in params else 13
        self.num_heads = params["num_heads"] if "num_heads" in params else 10
        self.input_embedding_dim = params["input_embedding_dim"] if "input_embedding_dim" in params else 1024
        self.filter_sizes = params['filter_sizes']
        self.filter_nums = params['filter_nums']
        self.epitope_filter_sizes = params['epitope_filter_sizes'] if params['epitope_filter_sizes'] else params['filter_sizes']
        self.conv_2_filter_size = params['conv_2_filter_size']
        self.num_tf_blocks = params['num_tf_blocks']
        # self.batch_size = params['batch_size']
        self.pos_encoding = params['pos_encoding']
        self.dropout = params['dropout'] if 'dropout' in params else 0.0
        self.dropout_attn = params['dropout_attn'] if 'dropout_attn' in params else 0.0
        self.dropout2 = params['dropout2'] if 'dropout2' in params else 0.0
        # self.output_vj = params['output_vj']
        self.output_vj = not params["ignore_vj"]
        # self.output_mhc = params['output_mhc']
        self.output_mhc = not params["ignore_mhc"]

        self.only_beta = params["only_beta"]
        self.only_alpha = params["only_alpha"] if "only_alpha" in params else False
        self.mhc_hi = params["mhc_hi"]
        self.old_ab = False
        self.feed_forward = params['feed_forward'] if 'feed_forward' in params else False
        assert(sum(self.filter_nums) % self.num_heads ==0)
        assert  not (self.only_alpha and self.only_beta), "Cannot set only_ for both chains, omit/set to False 'only_alpha' and 'only_beta' if you want to use both chains (recommended) or use either one."

        self.TCR_conv = nn.ModuleList(
            [nn.Conv1d(self.input_embedding_dim,self.filter_nums[i],k,padding=(k-min(self.filter_sizes))//2 + ( min(self.filter_sizes)//2 if self.feed_forward else 0) )  for i,k in enumerate( self.filter_sizes) ]
        )
        temp = ([nn.BatchNorm1d(sum(self.filter_nums)),nn.ReLU()] if params['BN'] else [ nn.ReLU()] ) + ([nn.Conv1d(sum(self.filter_nums),sum(self.filter_nums),self.conv_2_filter_size,padding=(self.conv_2_filter_size-1)//2)] if self.conv_2_filter_size >0 else []) +[nn.Dropout(self.dropout)]
        self.TCR_conv_2 = nn.Sequential(*temp) 

        

        self.epitope_conv = nn.ModuleList(
            [nn.Conv1d(self.input_embedding_dim,self.filter_nums[i],k,padding=(k-min(self.epitope_filter_sizes))//2 + ( min(self.epitope_filter_sizes)//2 if self.feed_forward else 0))  for i,k in enumerate( self.epitope_filter_sizes) ]
        )

        self.epitope_conv_2 = nn.Sequential(*copy.deepcopy(temp))
        # self.epitope_conv = nn.ModuleList(
        #     [nn.Conv1d(self.input_embedding_dim,30,k,padding=(k-3)//2 ) for k in [3,5,7] ]
        # )
        embed_dim = 0
        if self.output_vj:
            
            self.v_embed = nn.Embedding(166,8)
            self.j_embed = nn.Embedding(30,8)
            if self.only_alpha:
                self.v_embed = nn.Embedding(162,8)
                self.j_embed = nn.Embedding(112,8)
            elif not self.only_beta:
                self.av_embed = nn.Embedding(162,8)
                self.aj_embed = nn.Embedding(112,8)
            embed_dim = embed_dim + 16
        
        if self.output_mhc:
            if self.mhc_hi:
                self.mhc_embed = nn.ModuleList([nn.Embedding(3,2,0),nn.Embedding(5,4,0),nn.Embedding(params["MHC_lvl2_dim"],6,0),nn.Embedding(params["MHC_lvl3_dim"],8,0)]) #43 49
                embed_dim = embed_dim +20
            else:
                self.mhc_embed = nn.Embedding(params["MHC_all_dim"],8,0) #130
                embed_dim = embed_dim +8

        # self.tf_in = sum(self.filter_nums) * (self.TCR_max_length+self.epitope_max_length-2*min(self.filter_sizes)+2)

        #feed forward type of dense layer (acting on input dimension)
        self.ff_out_dim=50
        if self.feed_forward:
            self.ff_layer = nn.Sequential(nn.Linear(self.input_embedding_dim,self.ff_out_dim),nn.ReLU())
        # embedding dimension of transformer blocks
        tf_embedding_dim = sum(self.filter_nums) if not self.feed_forward else (sum(self.filter_nums) + self.ff_out_dim)
        # length of beta +epi (alpha +epi) after convolution 
        post_conv_TCR_epi_len = (self.TCR_max_length-min(self.filter_sizes)+1+self.epitope_max_length-min(self.epitope_filter_sizes)+1) if not self.feed_forward else (self.TCR_max_length+ self.epitope_max_length)
        #flattened in/out of transformer block = tf_embedding_dim * post_conv_TCR_epi_len
        self.tf_in = tf_embedding_dim * post_conv_TCR_epi_len


        
        self.skip_out = int(np.sqrt(sum(self.filter_nums)*(self.TCR_max_length + self.epitope_max_length))) if params['skip'] else 0
        
        self.fc_in = self.tf_in + self.skip_out + embed_dim
        self.skip_in =self.input_embedding_dim*(self.TCR_max_length+self.epitope_max_length)
        
        # self.mh = nn.MultiheadAttention(sum(self.filter_nums),self.num_heads)
        # self.ln = nn.LayerNorm(sum(self.filter_nums))
        if self.pos_encoding:
            self.positional_encoding= nn.parameter.Parameter(torch.empty((tf_embedding_dim,post_conv_TCR_epi_len)).normal_())
            if not (self.only_beta or self.only_alpha):
                self.positional_encoding_a= nn.parameter.Parameter(torch.empty((tf_embedding_dim,post_conv_TCR_epi_len)).normal_())
        
        self.tf_blocks = nn.Sequential(*([TransformerBlock(tf_embedding_dim,self.num_heads,n_hidden=int(1.5*tf_embedding_dim),dropout=self.dropout_attn) for _ in range(self.num_tf_blocks-1)] + [TransformerBlock(tf_embedding_dim,self.num_heads,output_attn_w=True,n_hidden=int(1.5*tf_embedding_dim),dropout=self.dropout_attn)] if self.num_tf_blocks >0 else [ SelfAttention(tf_embedding_dim,self.num_heads,dropout=self.dropout_attn)] ))
        
        self.skip = nn.Sequential(*(nn.Linear(self.skip_in,self.skip_out),nn.ReLU(),nn.Dropout(self.dropout)) if params['skip'] else [nn.Identity()])
        
        self.fc = nn.Sequential(nn.Linear(self.fc_in,int(np.sqrt(self.fc_in))),nn.Dropout(self.dropout2),nn.ReLU())

        self.fc_beta_only = nn.Sequential(nn.Linear(int(np.sqrt(self.fc_in)),1),nn.Sigmoid())

        if not (self.only_beta or self.only_alpha):
            self.alpha_conv = nn.ModuleList(
                [nn.Conv1d(self.input_embedding_dim,self.filter_nums[i],k,padding=(k-min(self.filter_sizes))//2 + ( min(self.filter_sizes)//2 if self.feed_forward else 0) )  for i,k in enumerate( self.filter_sizes) ]
            )
            self.alpha_conv_2 = nn.Sequential(*copy.deepcopy(temp)) 

            self.tf_blocks_a = nn.Sequential(*([TransformerBlock(tf_embedding_dim,self.num_heads,n_hidden=int(1.5*tf_embedding_dim),dropout=self.dropout_attn) for _ in range(self.num_tf_blocks-1)] + [TransformerBlock(tf_embedding_dim,self.num_heads,output_attn_w=True,n_hidden=int(1.5*tf_embedding_dim),dropout=self.dropout_attn)] if self.num_tf_blocks >0 else [ SelfAttention(tf_embedding_dim,self.num_heads,dropout=self.dropout_attn)] ))
            
            self.skip_a = nn.Sequential(*(nn.Linear(self.skip_in,self.skip_out),nn.ReLU(),nn.Dropout(self.dropout)) if params['skip'] else [nn.Identity()])
            
            self.fc_a = nn.Sequential(nn.Linear(self.fc_in,int(np.sqrt(self.fc_in))),nn.Dropout(self.dropout2),nn.ReLU())

            self.fc_alpha_only = nn.Sequential(nn.Linear(int(np.sqrt(self.fc_in)),1),nn.Sigmoid())

            self.fc_combine = nn.Sequential(nn.Linear(2*int((np.sqrt(self.fc_in))),int(2*int((np.sqrt(self.fc_in)))/3) ) ,nn.Dropout(self.dropout2),nn.ReLU() , nn.Linear(int(2*int((np.sqrt(self.fc_in)))/3) ,1) ,nn.Sigmoid() ) 



        
        
        
    def forward(self,TCR_in,Epitope_in,Alpha_in=None,V=None,J=None,TCR_len_in=None,aV=None,aJ=None,alpha_len_in=None,MHC_A=None,MHC_class=None,epi_len=None):
        TCR_len = TCR_len_in
        alpha_len = alpha_len_in
        
        betas = TCR_len > 0
        alphas  = alpha_len >0
        Epitope = torch.cat([conv(Epitope_in) for conv in self.epitope_conv],dim=1)
        assert Epitope.isnan().sum()==0
        Epitope = self.epitope_conv_2(Epitope)
        assert Epitope.isnan().sum()==0
        #(batch,embedding dim,seq_len)
        cdr3s = alphas if self.only_alpha else betas
        if self.only_alpha:
            TCR_in =Alpha_in
            J=aJ
            V=aV

        TCR = torch.cat([conv(TCR_in[cdr3s]) for conv in self.TCR_conv],dim=1)
        TCR = self.TCR_conv_2(TCR)
        y = torch.cat([TCR,Epitope[cdr3s] ],dim=2)

        if self.feed_forward:
            ffd_epi = self.ff_layer(Epitope_in.permute(0,2,1)).permute(0,2,1)
            ffd_beta = self.ff_layer(TCR_in.permute(0,2,1)).permute(0,2,1)
            y = torch.cat([y,torch.cat([ffd_beta,ffd_epi],dim=2)],dim=1)
        if len(y) > 0:
            if self.pos_encoding:
                y = y + self.positional_encoding
            y=y.permute(2,0,1)
            y,_ = self.tf_blocks(y)
            y = y.permute(1,2,0).reshape(-1,self.tf_in)

            if self.skip_out >0:
                sk = self.skip(torch.cat([TCR_in[cdr3s],Epitope_in[cdr3s]],dim=2).view(-1,self.skip_in))  
                y = torch.cat([y,sk],dim=1)
            if self.output_vj:
                y = torch.cat([y,self.v_embed(V[cdr3s]).view(-1,8),self.j_embed(J[cdr3s]).view(-1,8)],dim=1)
            if self.output_mhc:
                if self.mhc_hi:
                    y =torch.cat([y]+[self.mhc_embed[i](MHC_A[:,i][cdr3s]).view(-1,self.mhc_embed[i].weight.shape[1]) for i in range(4)],dim=1)
                else:
                    y = torch.cat([y,self.mhc_embed(MHC_A[cdr3s]).view(-1,8)],dim=1)

            y = self.fc(y) #.reshape(-1,self.fc_in)

        if self.only_beta or self.only_alpha or self.old_ab:
            y = self.fc_beta_only(y)
            if self.only_beta or self.only_alpha:
                return y

        Alpha = torch.cat([conv(Alpha_in[alphas]) for conv in self.alpha_conv],dim=1)
        Alpha = self.alpha_conv_2(Alpha)
        y2 = torch.cat([Alpha,Epitope[alphas]],dim=2)

        if self.feed_forward:
            
            ffd_alpha = self.ff_layer(Alpha_in[alphas].permute(0,2,1)).permute(0,2,1)
            y2 = torch.cat([y2,torch.cat([ffd_alpha,ffd_epi[alphas]],dim=2)],dim=1)

        if len(y2) >0:
            if self.pos_encoding:
                y2 = y2 + self.positional_encoding_a

            y2=y2.permute(2,0,1)
            # y,_ = self.mh(query=y,key=y,value=y)
            y2,_ = self.tf_blocks(y2)
            # y = self.ln(y)
            
            y2 = y2.permute(1,2,0).reshape(-1,self.tf_in)

            if self.skip_out >0:
                sk2 = self.skip(torch.cat([Alpha_in[alphas],Epitope_in[alphas]],dim=2).view(-1,self.skip_in))  
                y2 = torch.cat([y2,sk2],dim=1)

            if self.output_vj:
                y2 = torch.cat([y2,self.av_embed(aV[alphas]).view(-1,8),self.aj_embed(aJ[alphas]).view(-1,8)],dim=1)
            if self.output_mhc:
                if self.mhc_hi:
                    y2 =torch.cat([y2]+[self.mhc_embed[i](MHC_A[:,i][alphas]).view(-1,self.mhc_embed[i].weight.shape[1]) for i in range(4)],dim=1)
                else:
                    y2 = torch.cat([y2,self.mhc_embed(MHC_A[alphas]).view(-1,8)],dim=1)
            
            y2 = self.fc_a(y2)
            if self.old_ab:
                y2 = self.fc_alpha_only(y2)

        ret = torch.zeros(TCR_in.shape[0],1,device=y.device)
        
        if self.old_ab:
            if len(y) >0:
                ret[betas] = y
            if len(y2) >0:
                ret[alphas] +=y2
            div = ((TCR_len >0).int() + (alpha_len > 0).int()).view(ret.shape).to(y.device)
            
            ret = ret/ div
        else:
            alphasbetas = alphas & betas
            if alphasbetas.sum() >0:
                y3 = self.fc_combine(torch.cat([y[alphasbetas[betas]],y2[alphasbetas[alphas]]],dim=1))
                ret[alphasbetas] = y3
            if betas.sum()> 0:
                z = self.fc_beta_only(y[(betas & ~alphasbetas)[betas]] )
            
                ret[betas & ~alphasbetas] = z
            if alphas.sum()> 0:
                z2 = self.fc_alpha_only(y2[(alphas & ~alphasbetas)[alphas]] )
            
                ret[alphas & ~alphasbetas] =z2
        assert ret.isnan().sum()==0
        return ret

class LitEPICTRACE(pl.LightningModule):
   
    def __init__(self,hparams,model_class=EPICTRACE):
        super().__init__()
        self.load_MHC_dicts = not hparams.get("use_hardcoded_MHC_dicts",False)
        self.mhc_dict_path = hparams.get("MHC_dict","data/MHC_all_dict.bin")
        if self.load_MHC_dicts:
            with open("data/MHC_lvl2nd_dict.bin","rb") as handle:
                hparams["MHC_lvl2_dim"] = len(pickle.load(handle))
            with open("data/MHC_lvl3rd_dict.bin","rb") as handle:
                hparams["MHC_lvl3_dim"] = len(pickle.load(handle))
            
            with open(self.mhc_dict_path,"rb") as handle:
                hparams["MHC_all_dim"] = len(pickle.load(handle))
        else:
            hparams["MHC_lvl2_dim"] = 43
            hparams["MHC_lvl3_dim"] = 49
            hparams["MHC_all_dim"] = 130

        self.EPICTRACE_model = model_class(hparams)
        self.lr = hparams['lr']
        self.batch_size = hparams['batch_size']
        self.dataset = hparams['dataset']
        self.num_cpus = hparams['cpus'] if hparams['cpus']  else len(os.sched_getaffinity(0))
        print("Using ", self.num_cpus, "CPUs")
        self.input_embedding= True if hparams['input_embedding_data'] else False
        self.collate = hparams["collate"]
        self.wd = hparams["wd"]
        self.test_task = hparams["test_task"]
        self.lr_s = hparams["lr_s"] if "lr_s" in hparams else True
        assert self.test_task in [-1,0,1,2,3]
        self.add_01 = hparams["add_01"] if "add_01" in hparams else False
        self.only_CDR3 = hparams["only_CDR3"] if "only_CDR3" in hparams else False
        self.train_datapath = hparams["train"] if (hparams["train"] is not None) else os.getcwd() + '/data/'+self.dataset+'_train_data' 
        self.val_datapath = hparams["val"] if (hparams["val"] is not None) else os.getcwd() + '/data/'+self.dataset+'_validate_data'
        self.test_datapath = hparams["test"] if (hparams["test"] is not None) else (os.getcwd() + '/data/'+self.dataset+'_tpp'+ str(self.test_task) +'_data' if (self.test_task > -1) else os.getcwd() + '/data/'+self.dataset+'_test_data')
        self.version = hparams["version"]

        self.ed = None
        if self.input_embedding:
            # print("debugging not actually loading embedding")
            self.input_embedding_data = hparams['input_embedding_data']
            
            
        


        # weight assumed to be (label*(4.0) + 1)/epitope_freq but can be choosen freely using hp "linear"
        if not hparams["weight_fun"]:
            def _weight_fun(label,weight):
                return label*(4.0) + 1
            
        elif hparams["weight_fun"] == "linear":
            def _weight_fun(label,weight):
                return weight
            
        elif hparams["weight_fun"] =="log":
            def _weight_fun(label,weight):
                return (label*(4.0) + 1)/( -torch.log( weight /(label*(4.0) + 1) )+1)

        self.weight_fun=_weight_fun

        self.save_hyperparameters()

    def load_embedding_dict(self):
        # TODO: when embdict required and not available create, save and load
        if self.ed is None:
            if self.input_embedding:
                # print("debugging not actually loading embedding")
                
                with open(os.getcwd() +'/data/' +self.input_embedding_data , 'rb') as handle:
                    self.ed = pickle.load(handle)
            else:
                if "embedding" in self.collate:
                    # use os to run get_embs.py for the datasets
                    # 1 create joint dataset train+val+test
                        pd.concat([pd.read_csv(d) for d in [self.train_datapath,self.val_datapath,self.test_datapath] if os.path.exists(d)]).to_csv("data/joint_data.csv.gz",index=False)
                    # 2 run get_embs.py
                    # 2.1 change to correct directory
                        os.chdir("protBERT")

                    # Run get_embs.py using the Python interpreter from the desired virtual environment
                        subprocess.run([
                            "protbert_venv/bin/python",
                            "get_embs.py",
                            "../data/joint_data.csv.gz",
                            str(self.version)
                        ])
                    
                    # 3 load the embedding dict
                        with open(f"{self.version}allT.bin","rb") as handle:
                            self.ed = pickle.load(handle)
                    #  4 Clean up
                        os.system("rm ../data/joint_data.csv.gz")
                        # os.system(f"rm {self.version}allT.bin")
                        os.chdir("..")
                        
        return self.ed
    
    def forward(self,TCR,Vs,Js,TCR_len ,alpha, aVs,aJs,alpha_len,MHC_As,MHC_class, epitope,epi_len,label=None,weight=None):
        pred = self.EPICTRACE_model(TCR,epitope,alpha,Vs,Js,TCR_len,aVs,aJs,alpha_len,MHC_As,MHC_class,epi_len)
        return pred
        
    def step(self,batch):
        # TCR,epitope,TCR_len,epi_len,V,J,MHC_A , label= batch
        TCR,Vs,Js,TCR_len ,alpha, aVs,aJs,alpha_len,MHC_As,MHC_class, epitope,epi_len,label,weight = batch
        pred = self.EPICTRACE_model(TCR,epitope,alpha,Vs,Js,TCR_len,aVs,aJs,alpha_len,MHC_As,MHC_class,epi_len)
        return pred,label,weight

    def predict_step(self, batch, batch_idx):

        return self(batch)

    def training_step(self,batch,batch_idx):
        pred ,label,weight = self.step(batch)
        loss = F.binary_cross_entropy(pred.view(-1),label.to(torch.float),weight= self.weight_fun(label,weight))
        self.log('train_loss', loss)
        #loss = torch.tensor(0.0,requires_grad=True)
        return loss

    def validation_step(self,batch,batch_idx,dataloader_idx=None):
        
        pred ,label,weight = self.step(batch)
        # pred = torch.tensor([0.0,0.0]).view(-1)
        # label = torch.tensor([0,1],dtype=torch.int32).view(-1)
        # weight = torch.tensor([0.0,0.0]).view(-1)
        
        loss = F.binary_cross_entropy(pred.view(-1),label.to(torch.float),weight=self.weight_fun(label,weight))
        self.log('valid_loss', loss)
        ret = {
            'pred' : pred.view(-1),
            'label' : label,
            'valid_loss' : loss
        }
        return ret

    def validation_epoch_end(self,outputs):
        # avg_loss = torch.stack([x["valid_loss"] for x in outputs[0]]).mean()
        # preds = torch.cat([ x['pred'] for x in outputs[0]])
        # labels = torch.cat([ x['label'] for x in outputs[0]])
        
        # roc_auc = torchmetrics.functional.auroc(preds,labels)
        # ap = torchmetrics.functional.average_precision(preds,labels)
        
       
        # print('ROC_AUC',roc_auc)
        # print('ap',ap)
        # self.log('valid_ap',ap)
        # self.log('valid_ROC_AUC',roc_auc)
        # self.log('valid_avg_loss',avg_loss)


        # avg_loss = torch.stack([x["valid_loss"] for x in outputs[1]]).mean()
        # preds = torch.cat([ x['pred'] for x in outputs[1]])
        # labels = torch.cat([ x['label'] for x in outputs[1]])

        # roc_auc = torchmetrics.functional.auroc(preds,labels)
        # ap = torchmetrics.functional.average_precision(preds,labels)
        
       

        # print('ROC_AUC_test',roc_auc)
        # print('ap_test',ap)
        # self.log('test_ap',ap)
        # self.log('test_ROC_AUC',roc_auc)
        # self.log('test_avg_loss',avg_loss)


        # print(outputs)
        se = ['valid', 'valid2','valid3','valid_mix']
        if type(outputs[0]) == type(dict()):
            outputs=[outputs]
        for idx, out in enumerate(outputs):
            avg_loss = torch.stack([x["valid_loss"] for x in out]).mean()
            preds = torch.cat([ x['pred'] for x in out])
            labels = torch.cat([ x['label'] for x in out])
            
            roc_auc = torchmetrics.functional.auroc(preds,labels)
            ap = torchmetrics.functional.average_precision(preds,labels)
            # roc_auc = torch.tensor(0.0)
            # ap = torch.tensor(0.0)
        
            # print('ROC_AUC_' + se[idx],roc_auc)
            # print('ap_'+ se[idx],ap)
            self.log(se[idx] + '_ap',ap)
            self.log(se[idx] + '_ROC_AUC',roc_auc)
            self.log(se[idx] + '_avg_loss',avg_loss)

    def test_step(self,batch,batch_idx,dataloader_idx=None):
        pred ,label,weight = self.step(batch)

        loss = F.binary_cross_entropy(pred.view(-1),label.to(torch.float),weight=self.weight_fun(label,weight))
        self.log('test_loss', loss)
        ret = {
            'pred' : pred.view(-1),
            'label' : copy.deepcopy(label),
            'test_loss' : loss
        }
        del label
        return ret
    def test_epoch_end(self,outputs):
        se = ['TPP1','TPP2','TPP3','TMIX']
        
        # if test_task [0,1,2,3] then outputs = [test_step1,test_step2...] on given task
        # if test_task == -1 then outputs = [dataloader1_test_steps...] i.e. list of lists of steps for all 4 dataloaders
        if self.test_task != -1 :
            se[0]=se[self.test_task]
            outputs = [outputs]
        
        for idx, out in enumerate(outputs):
            avg_loss = torch.stack([x["test_loss"] for x in out]).mean()
            preds = torch.cat([ x['pred'] for x in out])
            labels = torch.cat([ x['label'] for x in out])
            roc_auc = torchmetrics.functional.auroc(preds,labels)
            ap = torchmetrics.functional.average_precision(preds,labels)
            
        
            print('ROC_AUC_' + se[idx],roc_auc)
            print('ap_'+ se[idx],ap)
            self.log(se[idx] + '_ap',ap)
            self.log(se[idx] + '_ROC_AUC',roc_auc)
            self.log(se[idx] + '_avg_loss',avg_loss)
        
        # def on_train_start(self):
    #     print("train start")
    #     self.logger.log_hyperparams(self.hparams)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,weight_decay= self.wd)

        if self.lr_s:
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience=6),
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer,0.9),
                "monitor": "valid_mix_ap"
                
                },
            }
        return optimizer

    def train_dataloader(self):
        
        
        tcr_train_data = PPIDataset2(csv_file = self.train_datapath,embedding_dict=self.load_embedding_dict(),
            output_vj=self.EPICTRACE_model.output_vj,output_mhc_A=self.EPICTRACE_model.output_mhc, mhc_hi=self.EPICTRACE_model.mhc_hi,add_01=self.add_01,only_CDR3=self.only_CDR3,load_MHC_dicts = self.load_MHC_dicts, mhc_dict_path=self.mhc_dict_path )
        
        tcr_train_data.TCR_max_length = self.EPICTRACE_model.TCR_max_length
        tcr_train_data.epitope_max_length = self.EPICTRACE_model.epitope_max_length


        train_dataloader = DataLoader(tcr_train_data, batch_size=self.batch_size, shuffle=True,
                                collate_fn=tcr_train_data.give_collate(self.collate) ,num_workers=self.num_cpus ,pin_memory=True)
        
        return train_dataloader

    def val_dataloader(self):
        
        
        
        tcr_test_data = PPIDataset2(csv_file = self.val_datapath,embedding_dict=self.load_embedding_dict(),
            output_vj=self.EPICTRACE_model.output_vj,output_mhc_A=self.EPICTRACE_model.output_mhc, mhc_hi=self.EPICTRACE_model.mhc_hi,add_01=self.add_01,only_CDR3=self.only_CDR3,load_MHC_dicts = self.load_MHC_dicts, mhc_dict_path=self.mhc_dict_path)
        tcr_test_data.TCR_max_length = self.EPICTRACE_model.TCR_max_length
        tcr_test_data.epitope_max_length = self.EPICTRACE_model.epitope_max_length

        
        test_dataloader = DataLoader(tcr_test_data, batch_size=self.batch_size,
                                collate_fn=tcr_test_data.give_collate(self.collate) ,num_workers=self.num_cpus ,pin_memory=True)
            
        
        return test_dataloader

    def test_dataloader(self,path=None,emb_dict=None):
        path = path if path else self.test_datapath
        emb_dict = emb_dict if emb_dict else self.load_embedding_dict()
        

        tcr_test_data = PPIDataset2(csv_file = path,embedding_dict=emb_dict,
            output_vj=self.EPICTRACE_model.output_vj,output_mhc_A=self.EPICTRACE_model.output_mhc, mhc_hi=self.EPICTRACE_model.mhc_hi,add_01=self.add_01,only_CDR3=self.only_CDR3,load_MHC_dicts = self.load_MHC_dicts, mhc_dict_path=self.mhc_dict_path)
        tcr_test_data.TCR_max_length = self.EPICTRACE_model.TCR_max_length
        tcr_test_data.epitope_max_length = self.EPICTRACE_model.epitope_max_length

        
        test_dataloader = DataLoader(tcr_test_data, batch_size=self.batch_size,
                                collate_fn=tcr_test_data.give_collate(self.collate)  ,num_workers=self.num_cpus ,pin_memory=True)
            
            
        
        return test_dataloader
