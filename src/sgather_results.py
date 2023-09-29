
import pickle
import pandas as pd
import os
import glob

def gather_fold_res():
    res_df = pd.read_csv("results/results.csv",index_col=0) if os.path.exists("results/results.csv") else pd.DataFrame()
    saved = []
    for dic in os.listdir("results/result_dicts"):
        print(dic)
        with open("results/result_dicts/"+dic,"rb") as handle:
            di = pickle.load(handle)
        res_df = pd.concat([res_df,pd.DataFrame(di,index=[dic[:dic.find(".bin")]])])
        saved.append("results/result_dicts/"+dic)
    res_df.to_csv("results/results.csv")
    for file in saved:
        os.remove(file)





def gather_PEpi_res():
    for stat in ["ap","auroc","count"]:

        pathname = "results/resall_PEpi_TPP2_" + stat + ".csv" 
        
        PEpires_df = pd.read_csv(pathname,index_col=0)  if os.path.exists(pathname) else pd.DataFrame()
        saved = []
        for ser_file in glob.glob("results/PEpi_result_series/*" +stat +"*"):
            print(ser_file)
            
            ser = pd.read_pickle(ser_file)
            PEpires_df = pd.concat([PEpires_df,pd.DataFrame(ser).T])
            saved.append(ser_file)
        PEpires_df.to_csv("results/resall_PEpi_TPP2_" + stat + ".csv")
        for ser_file in saved:
            os.remove(ser_file)
if __name__ == '__main__':
    gather_fold_res()
    gather_PEpi_res()