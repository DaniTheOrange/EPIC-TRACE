

files = ["outputs22/slurm_66489920.out","outputs22/slurm_66652857.out","outputs22/slurm_66660117.out","outputs22/slurm_66667338.out", "outputs22/slurm_66671670.out"]
def get_result_matrix(files):
    with open('outF.txt','w') as f2:
        f2.write("\n".join(files))
        for file in files:
            with open(file,'r') as f:
                lines = f.readlines()
                
                counter=0
                skip=False
                for line in lines:
                    if line.__contains__("DATALOADER"):
                        if line.__contains__("DATALOADER:0"):
                        
                            skip=False
                        else:
                            skip=True
                    if not skip:
                        fauc = line.find("AUC':")
                        if fauc != -1:
                            f2.write( line[fauc+6:line.find(",")] +"\t")
                            counter +=1

                        fauc = line.find("ap':")
                        if fauc != -1:
                            f2.write( line[fauc+5:line.find(",")]+"\t")
                            counter +=1
                        if counter ==8:
                            counter=0
                            f2.write("\n")
get_result_matrix(files)

