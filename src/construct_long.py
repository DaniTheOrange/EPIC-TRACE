import numpy as np
import csv
import re
import pandas as pd
import os
from src.split import set_diff
# copied from corresponding jupyter notebook
def add_gap(tcr,l_max,gap_char='-'):
    """Add gap to given TCR. Returned tcr will have length l_max.
    If there is an odd number of letters in the sequence, one more letter is placed in the beginning."""  
    l = len(tcr)
    if l<l_max:
        i_gap=np.int32(np.ceil(l/2))
        tcr = tcr[0:i_gap] + (gap_char*(l_max-l))+tcr[i_gap:l]
    return tcr

def get_long(vb,jb,cdr3b,vbseqs,jbseqs,aligned=False,cdr3length=25):
    if aligned:
        cdr3b = add_gap(cdr3b,cdr3length,'.')
        long = vbseqs[vb]+cdr3b+jbseqs[jb]
    else:
        long = vbseqs[vb]+cdr3b+jbseqs[jb]
        long = long.replace('.','')
    return long

def file2dict_2(filename,key_fields,store_fields,delimiter='\t'):
    """Read file to a dictionary.
    key_fields: fields to be used as keys
    store_fields: fields to be saved as a list
    delimiter: delimiter used in the given file."""
    dictionary={}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile,delimiter=delimiter)    
        for row in reader:
            keys = [row[k] for k in key_fields]
            aas=[row[s] for s in store_fields]
            aas2=[a if a!=None else '' for a in aas]
            store= ''.join(aas2)
                
            sub_dict = dictionary
            for key in keys[:-1]:
                if key not in sub_dict: 
                    sub_dict[key] = {}
                sub_dict = sub_dict[key]
            key = keys[-1]
            if key not in sub_dict:
                sub_dict[key] = []
            sub_dict[key]=store
    return dictionary

vbseqs=file2dict_2('data/trbvs.tsv',['Allele'],[str(i) for i in range(1,104)],delimiter='\t')
jbseqs=file2dict_2('data/trbjs.tsv',['Allele'],[str(i) for i in range(1,17)],delimiter='\t')
for jb in jbseqs: # Trim part that overlaps with CDR3B
    jbseq = jbseqs[jb]
    jbseqs[jb] = jbseq[re.search(r'[FW]G.G',jbseq).start()+1:]
vaseqs=file2dict_2('data/travs.tsv',['Allele'],[str(i) for i in range(1,104)],delimiter='\t')
jaseqs=file2dict_2('data/trajs.tsv',['Allele'],[str(i) for i in range(1,22)],delimiter='\t')
remove =[]
for ja in jaseqs: # Trim part that overlaps with CDR3A
    try:
        jaseq = jaseqs[ja]
        jaseqs[ja] = jaseq[re.search(r'[FW]G.G',jaseq).start()+1:]
    except: # remove noncanonical seqs at least for now
        remove.append(ja)
for ja in remove:
    del(jaseqs[ja])

# check that everything looks ok
    # print(vbseqs['TRBV2*01'])
    # print(jbseqs['TRBJ1-1*01'])
    # print(vaseqs['TRAV1-1*01'])
    # print(jaseqs['TRAJ3*01'])

#### End of Emmis code 

# Get IEDB data

# alpha beta 

# dicts for correct V and J genes
# V (beta)
vgenes=pd.read_csv('data/trbvs.tsv',delimiter='\t')
vgeness = set(vgenes.Gene)
valleless = set(vgenes.Allele)
do_mapv =vgenes[['Gene','Allele']][~vgenes.Gene.duplicated(keep=False)]
vmap= dict(zip(do_mapv.Gene,do_mapv.Allele))
# V (alpha)
avgenes=pd.read_csv('data/travs.tsv',delimiter='\t')
avgeness = set(avgenes.Gene)
avalleless = set(avgenes.Allele)
do_mapv =avgenes[['Gene','Allele']][~avgenes.Gene.duplicated(keep=False)]
avmap= dict(zip(do_mapv.Gene,do_mapv.Allele))

# J (beta)
jgenes=pd.read_csv('data/trbjs.tsv',delimiter='\t')
jgeness = set(jgenes.Gene)
jalleless = set(jgenes.Allele)
do_mapj =jgenes[['Gene','Allele']][~jgenes.Gene.duplicated(keep=False)]
jmap= dict(zip(do_mapj.Gene,do_mapj.Allele))

# J (alpha)
ajgenes=pd.read_csv('data/trajs.tsv',delimiter='\t')
ajgeness = set(ajgenes.Gene)
ajalleless = set(ajgenes.Allele)
do_mapj =ajgenes[['Gene','Allele']][~ajgenes.Gene.duplicated(keep=False)]
ajmap= dict(zip(do_mapj.Gene,do_mapj.Allele))
del vgenes ,jgenes ,avgenes,ajgenes
# Functions for filtering IEDB data
def is_aa(seq):
    if  pd.notna(seq):
        assert type(seq)==str,seq
        for aa in seq:
            if aa not in list("GPAVLIMCFYWHKRQNEDST"):
                # print(aa)
                return False
    return True
def is_HLA(allele):
    
    return pd.isna(allele) or ('HLA' in allele)  
def pick_MHC(allele):
    if ',' in allele:
        return allele.split(',')[0]
    else:
        return allele
def pick_MHC2(allele):
    idx = allele.find('HLA')
    idx2 = allele.find(':',idx +1)
    if idx2 == -1:
        return allele[idx:]
    else:
        return allele[idx:idx2+3]
def add_C(CDR3):
    if CDR3[0] != 'C':
        return 'C'+CDR3
    else:
        return CDR3
def check_ends(CDR3):
    if pd.notna(CDR3):
        assert type(CDR3)==str,CDR3
        if CDR3[0] == 'C':
            if not (CDR3[-1] =='W' or  CDR3[-1] =='F'):
                print("hmm not coherent CN...NNW/F  " +CDR3[-1] )
                return CDR3
            return CDR3[1:-1]
        
    return CDR3
    
        
def create_long(row):
    if not pd.notna(row['CDR3']):
        return row['CDR3']
    try:
        return get_long(row['V'],row['J'],row['CDR3'],vbseqs,jbseqs)
    except KeyError:
        # print(row)
        return row['CDR3']
def create_along(row):
    if not pd.notna(row['alpha']):
        return row['alpha']
    try:
        return get_long(row['aV'],row['aJ'],row['alpha'],vaseqs,jaseqs)
    except KeyError:
        # print(row)
        return row['alpha']
def correct_v(V):
    ret = V
    if  pd.notna(V):
        assert type(V)==str,V
        
        pos = ret.find(":")
        if pos != -1:
            pos2 =ret.find("-")
            
            if pos2 == -1:
                # print("replq: to -")
                # print(pos)
                # print(ret)
                ret = ret[:pos] +"-" +ret[pos+1:]
            pos = ret.find(":")
            if pos != -1:
                pos2 =ret.find("*")
                if pos2 == -1:
                    ret = ret[:pos] +"*" +ret[pos+1:]
        ret = ret.replace(" ","").replace("C","").replace("-0","-").replace("V0","V")
        if ("*" in ret) and (ret not in valleless):
            ret = ret[:ret.find("*")+3]
            if ret[:ret.find("-")] in vgeness:
                ret = ret[:ret.find("-")] +ret[ret.find("*"):]
        elif ret in vmap:
            return vmap[ret]
        
    return ret
def correct_j(J):
    ret = J
    if pd.notna(J):
        
        pos = ret.find(":")
        if pos != -1:
            pos2 =ret.find("-")
            if pos2 == -1:
                ret = ret[:pos] +"-" +ret[pos+1:]
            pos = ret.find(":")
            if pos != -1:
                pos2 =ret.find("*")
                if pos2 == -1:
                    ret = ret[:pos] +"*" +ret[pos+1:]
        ret = ret.replace(" ","").replace("C","").replace("-0","-").replace("J0","J")
        if ("*" in ret) and (ret not in jalleless):
            ret = ret[:ret.find("*")+3]
            if ret[:ret.find("-")] in jgeness:
                ret = ret[:ret.find("-")] +ret[ret.find("*"):]
        elif ret in jmap:
            return jmap[ret]
    
    return ret

def correct_av(V):
    ret = V
    if  pd.notna(V):
        assert type(V)==str,V
        pos = ret.find(":")
        if pos != -1:
            pos2 =ret.find("-")
            if pos2 == -1:
                ret = ret[:pos] +"-" +ret[pos+1:]
            pos = ret.find(":")
            if pos != -1:
                pos2 =ret.find("*")
                if pos2 == -1:
                    ret = ret[:pos] +"*" +ret[pos+1:]
        ret = ret.replace(" ","").replace("C","").replace("-0","-").replace("V0","V")
        if ("*" in ret) and (ret not in avalleless):
            ret = ret[:ret.find("*")+3]
            if ret[:ret.find("-")] in avgeness:
                ret = ret[:ret.find("-")] +ret[ret.find("*"):]
        elif ret in avmap:
            return avmap[ret]
        
    return ret
def correct_aj(J):
    ret = J
    if pd.notna(J):
        pos = ret.find(":")
        if pos != -1:
            pos2 =ret.find("-")
            if pos2 == -1:
                ret = ret[:pos] +"-" +ret[pos+1:]
            pos = ret.find(":")
            if pos != -1:
                pos2 =ret.find("*")
                if pos2 == -1:
                    ret = ret[:pos] +"*" +ret[pos+1:]
        ret = ret.replace(" ","").replace("C","").replace("-0","-").replace("J0","J")
        if "*" in ret:
            ret.replace("-1","")
        else:
            ret.replace("-1","*01")
        if ("*" in ret) and (ret not in ajalleless):
            ret = ret[:ret.find("*")+3]
            if ret[:ret.find("-")] in ajgeness:
                ret = ret[:ret.find("-")] +ret[ret.find("*"):]
        elif ret in ajmap:
            return ajmap[ret]
    
    return ret

def get_genefamily(gene):
    pos = gene.find("-")
    if pos==-1:
        return gene
    else:
        return gene[:pos]

ajgenes=set(pd.read_csv('data/trajs.tsv',delimiter='\t').Gene)
ajgenes.update(set(pd.read_csv('data/trajs.tsv',delimiter='\t').Gene.map(get_genefamily)))

avgenes=set(pd.read_csv('data/travs.tsv',delimiter='\t').Gene)
avgenes.update(set(pd.read_csv('data/travs.tsv',delimiter='\t').Gene.map(get_genefamily)))

jgenes=set(pd.read_csv('data/trbjs.tsv',delimiter='\t').Gene)
jgenes.update(set(pd.read_csv('data/trbjs.tsv',delimiter='\t').Gene.map(get_genefamily)))

vgenes=set(pd.read_csv('data/trbvs.tsv',delimiter='\t').Gene)
vgenes.update(set(pd.read_csv('data/trbvs.tsv',delimiter='\t').Gene.map(get_genefamily)))

def fix_tr(gene,typ):
    if pd.isna(gene):
        return gene
    ret = gene.replace(" ","").replace("C","").replace("-0","-")
    if ret[4] =="0":
        ret= ret[:4]+ ret[5:]
    pos = ret.find('*')
    if pos == -1:
        pos = len(gene)
    if typ == "aJ":
        if pos>0:
            if ret[:pos] in ajgenes:
                pass
            else:
                ret = ret.replace("-1","")
        if ret in ajmap:
            ret = ajmap[ret]
    elif typ == "aV":
        if pos>0:
            if ret[:pos] in avgenes:
                pass
            else:
                ret = ret.replace("-1","")
        if ret in avmap:
            ret = avmap[ret]
    elif typ == "J":
        if pos>0:
            if ret[:pos] in jgenes:
                pass
            else:
                ret = ret.replace("-1","")
        if ret in jmap:
            ret = jmap[ret]
    elif typ == "V":
        if pos>0:
            if ret[:pos] in vgenes:
                pass
            else:
                ret = ret.replace("-1","")
        if ret in vmap:
            ret = vmap[ret]
    
    
    return ret

def fix_alpha(x):
    if pd.notna(x["alpha"] ) and (pd.isna(x["aJ"]) or pd.isna(x["aV"]) ):
        # print(x)
        # assert False
        ret = x["alpha"]
        
    elif pd.notna(x["alpha"] ) and not ("TRAJ33" in x["aJ"] or "TRAJ38" in x["aJ"]):
        if x["alpha"][-1]!="F":
            ret = x["alpha"] + "F"
        else:
            ret = x["alpha"]
    elif pd.notna(x["alpha"] ):
        if x["alpha"][-1]!="W":
            ret = x["alpha"] + "W"
        else:
            ret = x["alpha"]
    else:
        return x["alpha"]
    if x["alpha"][0] !="C":
        ret = "C" +ret
    return ret

def fix_beta(x):
    if pd.notna(x["CDR3"] ) and (pd.isna(x["J"]) or pd.isna(x["V"]) ):
        # print(x)
        # assert False
        ret = x["CDR3"]
        
    elif pd.notna(x["CDR3"] ):
        if x["CDR3"][-1]!="F":
            ret = x["CDR3"] + "F"     
        else:
            ret = x["CDR3"]   
    else:
        return x["CDR3"]
        
    if x["CDR3"][0] !="C":
        ret = "C" +ret
    else:
        ret = x["CDR3"]
    return ret

def not_F(gene,typ):
    ret =gene
    if pd.isna(ret):
        return False
    pos = ret.find('*')
    if pos == -1:
        pos = len(gene)
    if typ == "aJ":
        if pos>0:
            if ret[:pos] in ajgenes:
                return False
            else:
                return True
                
    elif typ == "aV":
        if pos>0:
            if ret[:pos] in avgenes:
                return False
            else:
                return True
        
    elif typ == "J":
        if pos>0:
            if ret[:pos] in jgenes:
                return False
            else:
                return True
    elif typ == "V":
        if pos>0:
            if ret[:pos] in vgenes:
                return False
            else:
                return True
vj_dict = dict()

for gene in ["aJ","aV","bJ","bV"]:
    
    fi = set(pd.read_csv('data/tr' + gene.lower() +'s.tsv',delimiter='\t').Gene.map(get_genefamily))
    sec = set(pd.read_csv('data/tr' + gene.lower() +'s.tsv',delimiter='\t').Gene)
    trd = set(pd.read_csv('data/tr' + gene.lower() +'s.tsv',delimiter='\t').Allele)
    vj_dict[gene if not "b" in gene else gene[-1]] = set().union(fi,sec,trd)

def fix_IEDB_data(trust_cur_nona_HS_pos2,trust_cur_nona_HS_neg, return_broken=False):
    only_pos = pd.concat([trust_cur_nona_HS_pos2,trust_cur_nona_HS_neg,trust_cur_nona_HS_neg]).drop_duplicates(keep=False)
    only_pos=only_pos[only_pos.CDR3.apply(is_aa)]
    only_pos=only_pos[only_pos.Epitope.apply(is_aa)]
    fil = only_pos['MHC A'].apply(is_HLA)
    if return_broken:
        broken1 = only_pos[~fil]
        broken1['MHC A'] = broken1['MHC A'].apply(correct_HLA)
        broken1['V'] = broken1['V'].apply(correct_v)
        broken1['J'] = broken1['J'].apply(correct_j)
        broken1['aV'] = broken1['aV'].apply(correct_av)
        broken1['aJ'] = broken1['aJ'].apply(correct_aj)
        for gene in ["aJ","aV","J","V"]:
            broken1[gene] = broken1[gene].map( lambda x : x if x in vj_dict[gene] else None)
    only_pos= only_pos[fil]
    # only_pos['MHC A'] = only_pos['MHC A'].apply(pick_MHC2)
    
    only_pos['MHC A'] = only_pos['MHC A'].apply(correct_HLA)
    only_pos['V'] = only_pos['V'].apply(correct_v)
    only_pos['J'] = only_pos['J'].apply(correct_j)
    only_pos['aV'] = only_pos['aV'].apply(correct_av)
    only_pos['aJ'] = only_pos['aJ'].apply(correct_aj)

    for gene in ["aJ","aV","J","V"]:
        print(gene)
        # only_pos[gene] = only_pos[gene].apply(fix_tr,typ=gene)
        # only_pos = only_pos[~only_pos[gene].apply(not_F ,typ=gene)]
        filt = only_pos[gene].map( lambda x: x in vj_dict[gene] if pd.notna(x) else True)
        if return_broken:
            to_broken = only_pos[~filt]
            to_broken[gene] = None
            broken1 = pd.concat([broken1,to_broken])
        only_pos = only_pos[filt]

        # alphabeta["notF" + gene] =~alphabeta[gene].apply(not_F ,typ=gene)
        print(len(only_pos))
    only_pos['CDR3'] = only_pos.apply(fix_beta,axis=1)
    only_pos['Long'] = only_pos.apply(create_long,axis=1)
    


    only_pos=only_pos[only_pos.alpha.apply(is_aa)]
    only_pos['alpha'] = only_pos.apply(fix_alpha,axis=1)
    only_pos['aLong'] = only_pos.apply(create_along,axis=1)
    
    only_pos=only_pos.drop_duplicates()
    if return_broken and len(broken1) >0:
        broken1['CDR3'] = broken1.apply(fix_beta,axis=1)
        broken1['Long'] = broken1.apply(create_long,axis=1)
        broken1=broken1[broken1.alpha.apply(is_aa)]
        broken1['alpha'] = broken1.apply(fix_alpha,axis=1)
        broken1['aLong'] = broken1.apply(create_along,axis=1)
        broken1=broken1.drop_duplicates()

    print(len(only_pos))
    if return_broken:
        return only_pos,broken1
    return only_pos

def process_IEDB(in_csv_pos,in_csv_neg,return_broken=False):
    iedb_HS_neg = pd.read_csv(in_csv_neg)
    iedb_HS_pos2 = pd.read_csv(in_csv_pos)
    old_file=['MHC Allele Names','Description','Curated Chain 2 V Gene','Curated Chain 2 J Gene','Chain 2 CDR3 Curated','Curated Chain 1 V Gene','Curated Chain 1 J Gene','Chain 1 CDR3 Curated']
    old_file_calc = ['Calculated Chain 2 V Gene','Calculated Chain 2 J Gene', 'Chain 2 CDR3 Calculated','Calculated Chain 1 V Gene','Calculated Chain 1 J Gene', 'Chain 1 CDR3 Calculated']
    
    iedb_cur = iedb_HS_pos2[old_file].replace(['nan'], None)
    iedb_cur.columns= ['MHC A','Epitope','V','J','CDR3','aV','aJ','alpha']
    iedb_cal = iedb_HS_pos2[old_file_calc].replace(['nan'], None)
    iedb_cal.columns= ['V','J','CDR3','aV','aJ','alpha']
    trust_cur_HS_pos2 = iedb_cur.combine_first(iedb_cal)

    iedb_cur = iedb_HS_neg[old_file].replace(['nan'], None)
    iedb_cur.columns= ['MHC A','Epitope','V','J','CDR3','aV','aJ','alpha']
    iedb_cal = iedb_HS_neg[old_file_calc].replace(['nan'], None)
    iedb_cal.columns= ['V','J','CDR3','aV','aJ','alpha']
    trust_cur_HS_neg = iedb_cur.combine_first(iedb_cal)

    # alphabeta =trust_cur_HS_pos2.dropna(subset=['MHC A','Epitope','aV','aJ','alpha'])
    alpha_filt = trust_cur_HS_pos2[['MHC A','Epitope','aV','aJ','alpha']].notna().all(axis=1)
    alphabeta = trust_cur_HS_pos2[alpha_filt]
    beta_filt = trust_cur_HS_pos2[['MHC A','Epitope','V','J','CDR3']].notna().all(axis=1)
    # alphabeta = pd.concat([alphabeta,trust_cur_HS_pos2.dropna(subset=['MHC A','Epitope','V','J','CDR3'])]).drop_duplicates(ignore_index=True)
    alphabeta = pd.concat([alphabeta,trust_cur_HS_pos2[beta_filt]]).drop_duplicates(ignore_index=True)

    alphabeta_filter =((alphabeta.alpha.map(lambda x: pd.notna(x) and x[0] =='C' and  (x[-1]=='F' or x[-1]=='W')) | alphabeta.alpha.map(lambda x: pd.notna(x) and x[0] !='C' and not (x[-1]=='F' or x[-1]=='W')) | ~alphabeta.alpha.notna()  ) & (alphabeta.CDR3.map(lambda x: pd.notna(x) and x[0] =='C' and  (x[-1]=='F' or x[-1]=='W')  ) | alphabeta.CDR3.map(lambda x: pd.notna(x) and x[0] !='C' and  (x[-1]=='F' or x[-1]=='W')) | ~alphabeta.CDR3.notna()) )
    if return_broken:
        broken = pd.concat([trust_cur_HS_pos2[~alpha_filt].dropna(subset=['Epitope','alpha']),trust_cur_HS_pos2[~beta_filt].dropna(subset=['Epitope','CDR3'])]).drop_duplicates(ignore_index=True)
        broken = pd.concat([broken,alphabeta[~alphabeta_filter]])
    alphabeta=alphabeta[alphabeta_filter]
    alphabeta = fix_IEDB_data(alphabeta,trust_cur_HS_neg,return_broken=return_broken)
    if return_broken:
        fixed_broken = pd.concat(fix_IEDB_data(broken,pd.DataFrame(),return_broken=return_broken))
        correct_alphabeta = alphabeta[0]
        broken_alphabeta = pd.concat([alphabeta[1],fixed_broken])
        return correct_alphabeta, broken_alphabeta
    return alphabeta 

def IEDB_to_csv(in_csv_pos,in_csv_neg, out_csv):
    """
    in_csv_pos : exported IEDB "tcell receptor table" of POSITIVE samples
    in_csv_neg : exported IEDB "tcell receptor table" of NEGATIVE samples
    out_csv    : output file path

    filtetrs the IEDB data and applies required corrections.
    also creates the long sequences when possible, when not possible
    the long sequence equals CDR3

    NOTE    return order same as in input 
            only FIRST HLA is taken if many in same line (not both if available MHC II )
            also MHC II chain varies (the saved is not necessarily alpha (or beta))
            "half" alpha beta pairs may exist (e.g. beta contains all information but alpha has aV missing)
    """
    alphabeta = process_IEDB(in_csv_pos,in_csv_neg)

    alphabeta.to_csv(out_csv,index=False)

# Get VDJDB data

# alpha beta 
def process_vdjdb(tsv_path,return_broken=False,dropped_cols=["Epitope species","Epitope gene","Reference","Method","Meta", "CDR3fix"]):
    vd_ab = pd.read_csv(tsv_path,delimiter='\t').drop_duplicates(ignore_index=True)
    vd_ab.drop(columns=dropped_cols,inplace=True)

    vdj_ab = vd_ab[vd_ab["complex.id"] !=0]
    vdj_a =vd_ab[(vd_ab["complex.id"] ==0)& (vd_ab.Gene =="TRA")].copy()
    merge_on = ["complex.id","Epitope","MHC A","MHC B" ,"Species", "MHC class"]
    vdj_a.rename(columns=dict(zip(vdj_a.columns,[ n+"_alpha" if n not in merge_on else n for n in vdj_a.columns])) ,inplace=True)
    print(vdj_a.columns)
    vdj_b =vd_ab[(vd_ab["complex.id"] ==0)& (vd_ab.Gene =="TRB")]

    merged_ab = vdj_ab[vdj_ab.Gene == "TRB"].merge(vdj_ab[vdj_ab.Gene == "TRA"], on=merge_on, suffixes=['', '_alpha'])
    print(len(merged_ab))

    vdjdb_all = pd.concat([merged_ab,vdj_a,vdj_b],ignore_index=True)

    vdjdb_all.drop(columns=["Species", "complex.id","Gene","Gene_alpha"],inplace=True)

    print(vdjdb_all.columns)
    cols = {"CDR3_alpha" :"alpha","V_alpha" : "aV","J_alpha" : "aJ",}
    vdjdb_all.rename(columns=cols,inplace=True)
    print(vdjdb_all.columns)

    vdjdb_all['MHC A'] = vdjdb_all['MHC A'].apply(correct_HLA)

    vdjdb_all['V'] = vdjdb_all['V'].apply(correct_v)
    vdjdb_all['J'] = vdjdb_all['J'].apply(correct_j)
    vdjdb_all['aV'] = vdjdb_all['aV'].apply(correct_av)
    vdjdb_all['aJ'] = vdjdb_all['aJ'].apply(correct_aj)

    if return_broken:
        broken = pd.DataFrame()
    for gene in ["aJ","aV","J","V"]:
        print(gene)
        filt = vdjdb_all[gene].map( lambda x: x in vj_dict[gene] if pd.notna(x) else True)

        if return_broken: 
            to_broken = vdjdb_all[~filt]
            to_broken[gene] = None
            broken = pd.concat([broken,to_broken])
        vdjdb_all = vdjdb_all[filt]

    vdjdb_all['Long'] = vdjdb_all.apply(create_long,axis=1)
    vdjdb_all['aLong'] = vdjdb_all.apply(create_along,axis=1)

    if return_broken :
        if len(broken) >0:
            broken['Long'] = broken.apply(create_long,axis=1)
            broken['aLong'] = broken.apply(create_along,axis=1)
        return vdjdb_all,broken
    return vdjdb_all

def vdjdb_to_csv(tsv_path,csv_out):
    """
    tsv_path: tsv exported from vdjdb
    csv_out : path for created autput file

    takes VDJDB tsv and conacatenates alphas with corresponding beta sequnces (columns)
    and adds rows for alphas and betas without pairs (having empty values in beta or alpha columns respectively)
    output features 
    CDR3	V	J	MHC A	MHC B	MHC class	Epitope	alpha	aV	aJ Long aLong

    NOTE MHC B!!
    """
    vdjdb_all = process_vdjdb(tsv_path)
    
    vdjdb_all.to_csv(csv_out,index=False)

def correct_tolong(row):
    row=row.copy()
    cols=row.index
    if len(row['Long']) > len(row['CDR3']):
        pass
    else:
        vo = row["V"]
        jo = row["J"]
        if not "*" in row["V"]:
            row["V"] = vo +"*01"
        if not "*" in row["J"]:
            row["J"] = jo +"*01"
        newlong = create_long(row)
        if len(newlong) > len(row['CDR3']):
            row['Long']=newlong
            row['longed']=True
        else:
            row["V"] = vo
            row["J"] = jo
            row['longed']=False
            
    if pd.isna(row["alpha"]) or len(row['aLong']) > len(row['alpha']):
        pass
    else:
        vo = row["aV"]
        jo = row["aJ"]
        if not "*" in row["V"]:
            row["aV"] = vo +"*01"
        if not "*" in row["J"]:
            row["aJ"] = jo +"*01"
        newlong = create_along(row)
        if len(newlong) > len(row['alpha']):
            row['aLong']=newlong
            row['longed']=True
        else:
            row["aV"] = vo
            row["aJ"] = jo
            row['longed']=False

        print(row)
    return row[cols.append(pd.Index(["longed"]))]


def filter_sim_MHC(mhc_list):
    if type(mhc_list)== str:
        mhc_list = mhc_list.split(",")
    good= []
    for idx,mhc in enumerate(mhc_list):
        gt=True
        for mhc2 in mhc_list[idx+1:]+good:
            if mhc in mhc2:
                gt=False
                break
        if gt:
            good.append(mhc)
    return good

def clip_MHC(mhc):
    pos = mhc.find(":")
    pos2 = mhc.find(":",pos+1)
    if pos2 > -1:
        mhc = mhc[:pos2]
    return mhc

def correct_HLA(mhc):
    
    if pd.isna(mhc) or "HLA class" in mhc:
        return mhc
    #remove synonymous mutations (unneseccary information)
    mhc = clip_MHC(mhc).replace(" ","")
    # add star if missing
    if mhc.find("*") == -1:
        pos = mhc.find("-")
        while (pos < len(mhc) -1) and mhc[pos+1].isalpha():
            pos +=1
        mhc = mhc[:pos+1] +"*" +mhc[pos+1:]
    pos = mhc.find("*")
    pos2 = mhc.find(":")
    if pos2 == -1:
        if pos +2 == len(mhc):
            mhc = mhc[:pos+1]+"0" +mhc[pos+1:] 
    else:
        if pos +2 == pos2:
            mhc = mhc[:pos+1]+"0" +mhc[pos+1:]
    return mhc

def filter_sim_MHC(mhc_list):
    """ very bad naming !
    inputs a list of strings and outputs the list such that elements
     that are substrings of at least one other lements are removed"""
    if type(mhc_list)== str:
        mhc_list = mhc_list.split(",")
    good= []
    for idx,mhc in enumerate(mhc_list):
        gt=True
        for mhc2 in mhc_list[idx+1:]+good:
            if mhc in mhc2:
                gt=False
                break
        if gt:
            good.append(mhc)
    return good

def remove_equiv_duplicates(df,subset,on="MHC A",keep_na=False,return_filter=False):
    """removes less informative duplicates, datapoints where all other columns are equal wehreas 'on' column has an equvalent or more informative value 
    e.g.datapoint with  HLA-A*02 will be removed if a datapoint with HLA-A*02:01 exists (and all other values are equal)
    
    subset: list of strings of the columns that should be equal
    on: column that the equivalent duplicates are removed from
    keep_na: bool if True na values in column "on" are accepted if there is no duplicate on based on the subset columns
    return_filter: just for debugging """
    # print(len(df.dropna(subset=["MHC A"])))
    catedMHC = df[subset +[on]].dropna(subset=[on]).groupby(subset,dropna=False).apply(lambda x :  ",".join([str(i) for i in x[on]]) )
    print("cated MHC ",len(catedMHC))
    cated2 = catedMHC.map(filter_sim_MHC).reset_index()
    df2 = df.merge(cated2,on=subset).rename(columns={0:"uniq"+on})
    print("len df2",len(df2))
    filt = df2.apply(lambda row: row[on] in row["uniq"+on],axis=1)
    df2["filt"] = filt
    ret = df2[filt]

    if keep_na:
        na_filt = df[on].isna()
        # good_na = pd.concat([df[na_filt],df[~na_filt],df[~na_filt]]).drop_duplicates(subset,keep=False) # removes unecessary dups 
        good_na = set_diff(df[na_filt],df[~na_filt],similarity_subset=subset)
        ret = pd.concat([ret,good_na])

    if return_filter:
        return ret,filt,df2
    return ret

def filter_equiv_dups(df,**kvargs):
    """removes MHC V J aJ and aV with remove euiv_duplicates"""
    df = remove_equiv_duplicates(df,["Epitope","CDR3","alpha","V","J","aJ","aV"],**kvargs)
    for onni in ["V","J","aJ","aV"]:
        
        lits = ["Epitope","CDR3","alpha","V","J","aJ","aV","MHC A"]
        lits.remove(onni)
        df = remove_equiv_duplicates(df,lits,on=onni,keep_na=True)
    return df