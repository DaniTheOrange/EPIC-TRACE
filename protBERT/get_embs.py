from fileinput import filename
from bert_mdl import retrieve_model, extract_and_save_embeddings, compute_embs
import pandas as pd
import torch
import argparse
import os
import pickle
# include as demo vdj50 <-

def get_to_BERT(newdata,base_dict,CDR3=False):
    if CDR3:
        # bcdrset = set(embedded.CDR3)
        # li = list(set(newdata.CDR3).difference(bcdrset))
        # betas = pd.DataFrame({"cdr3":li,"long":li})
        # acdrset = set(embedded.alpha)
        # li = list(set(newdata.alpha).difference(acdrset))
        # alphas = pd.DataFrame({"cdr3":li,"long":li})
        lon = "CDR3"
        lona = "alpha"
    else:
        lon = "Long"
        lona = "aLong"
        # alonset = set(embedded.aLong)
        alphas = newdata[newdata[lona].map(lambda x : x not in base_dict)][["alpha",lona]]
        alphas.columns = ["cdr3","long"]
        # lonset = set(embedded.Long)
        betas = newdata[newdata[lon].map(lambda x : x not in base_dict)][["CDR3",lon]]
        betas.columns = ["cdr3","long"]
    # episet = set(embedded.Epitope)
    li = []
    for epi in set(newdata.Epitope):
        if epi not in base_dict:
            li.append(epi)
    # li = list(set(newdata.Epitope).difference(episet))

    tobert = pd.concat([betas,alphas,pd.DataFrame({"cdr3":li,"long":li})]).drop_duplicates().dropna()
    return tobert

print("get_embs.py greetings")
if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input_file",help="path to the input file with sequence columns 'long' and 'cdr3' or new data file (that can be directly inputted to EPIC-TRACE, base_dict must be used)")
    parser.add_argument("emb_name",help="name of the output dicts '<emb_name>_<idx>.bin")
    parser.add_argument("--seqs_per_file", type=int,default=24000)
    parser.add_argument("--base_dict",help="update the new embeddings to the base dict")
    parser.add_argument("--save_path",help="location of the final dict")
    parser.add_argument("--overwrite_base_dict",default=False,action="store_true")
    args = parser.parse_args()
    
    dict2 = dict()
    if args.base_dict is not None:
        try:
            with open(args.base_dict,'rb') as handle:
                dict2 = pickle.load(handle)
        except:
            print("base_dict not found initialising empty base dict")
            dict2 = dict()

    # if aplicable create input file
    inp_file_df = pd.read_csv(args.input_file)
    if ("long" in inp_file_df.columns) and ("cdr3" in inp_file_df.columns):
        file_to_bert = args.input_file
    else:
        tobert_df = get_to_BERT(inp_file_df,dict2)
        file_to_bert = "tmp_data_toBERT.csv"
        tobert_df.to_csv(file_to_bert)


    # # load the model
    
    model = retrieve_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("protBERTmodel retrieved")
    
    # extract and save some embeddings
    # extract_and_save_embeddings(model, data_f_n="../epitcr/to_bert_17_6_aLong.csv", sequence_col="long", cdr3_col="cdr3", seqs_per_file=24000, emb_name='ievdj_alongs_17_6',emb_folder="./")
    # extract_and_save_embeddings(model, data_f_n="../epitcr/data/ydisplaytest_tobert.csv", sequence_col="long", cdr3_col="cdr3", seqs_per_file=24000, emb_name='ydisplaytest',emb_folder="./")
    extract_and_save_embeddings(model, data_f_n=file_to_bert, sequence_col="long", cdr3_col="cdr3", seqs_per_file=args.seqs_per_file, emb_name=args.emb_name,emb_folder="./")
    # extract_and_save_embeddings(model, data_f_n="../epitcr/to_bert_17_6_alpha_mcpas_cdr3.csv", sequence_col="long", cdr3_col="cdr3", seqs_per_file=24000, emb_name='ievdj_a_mcpas_cdr3_17_6',emb_folder="./")
    print("finished creating BERT embeds")
    # extract_and_save_embeddings(model, data_f_n="../epitcr/vdjdb_to_embs", sequence_col="long", cdr3_col="cdr3", seqs_per_file=24000, emb_name='vdjdb_embs') # this should not work
    
    # exit(123)
    # # compute some more embeddings (in an "online" fashion)
    # data = pd.read_csv("vdj_human_unique_longs.csv", sep=",")
    # long, cdr3b = data["long"].values[:100], data["cdr3b"].values[:100]
    # print(compute_embs(model, long)[0].shape, compute_embs(model, long)[-1].shape)
    # print(compute_embs(model, long, cdr3b)[0].shape, compute_embs(model, long, cdr3b)[-1].shape)

    # # load fine-tuned bert and compute some embeddings with the loaded model. A checkpoint model will be available after
    # # fine-tuning the BERT model
    # best_checkpoint_path = "experiments/lightning_logs/version_17-9-2021--13-39-35/checkpoints/epoch=8-val_loss=1.51-val_acc=0.96.ckpt"
    # model = model.load_from_checkpoint(best_checkpoint_path)
    # model.eval()
    # model.to(device)
    # print(compute_embs(model, long)[0].shape, compute_embs(model, long)[-1].shape)
    # print(compute_embs(model, long, cdr3b)[0].shape, compute_embs(model, long, cdr3b)[-1].shape)

    emb_file = args.emb_name +"_0.bin"
    wrongdict = dict()
    i=0
    while(os.path.exists(emb_file)):

        with open(emb_file , "rb") as handle:
            wrongdict.update(pickle.load(handle))
        i+=1
        emb_file =args.emb_name +"_"+ str(i)+".bin"

    def correct_shape_save(savename,wrongdict,dict2):

        
        for key,value in wrongdict.items():
            if value.shape[0] ==1024:
                dict2[key] = value.T
                
            else:
                dict2[key] = value

        with open(savename,'wb') as handle:
            pickle.dump(dict2,handle)
        
    if args.overwrite_base_dict:
        savename = args.base_dict
    else:
        if args.save_path is None:
            savename = args.emb_name + "allT.bin"
        else:
            savename = args.save_path +args.emb_name + "allT.bin"
    correct_shape_save(savename,wrongdict,dict2)
