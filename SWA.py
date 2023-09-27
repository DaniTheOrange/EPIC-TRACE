import torch
from EPICTRACE_model import LitEPICTRACE , EPICTRACE
import os
import torch.nn.functional as F

def do_SWA(pl_model:LitEPICTRACE,lr_max,lr_min,cycle_len,epochs=10,ret_path_name=True,save_dir=None):
    print("manual SWA with params ",lr_max," ",lr_min," ",cycle_len," ",epochs)

    if save_dir:
        dir_path = save_dir
    else:
        dir_path =  os.getcwd() +'/loggingDir22/folder22/version_'+ str(pl_model.hparams["hparams"]['version']) +'/checkpoints/' 
        
        if not os.path.exists(dir_path):
            dir_path = os.getcwd() +'/loggingDir22/folder22/versions'+ str(pl_model.hparams["hparams"]['version'])[:3]+'_' + str(pl_model.hparams["hparams"]['version'])[-2] +  '/version_'+ str(pl_model.hparams["hparams"]['version']) +'/checkpoints/'
    assert os.path.exists(dir_path)

    model = pl_model.EPICTRACE_model
    model.train()
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr_max)
    # pl_model.configure_optimizers()["optimizer"]
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=lr_min,max_lr=lr_max,step_size_up=1,step_size_down=cycle_len-1,cycle_momentum=False)
    dataset = pl_model.train_dataloader()
    for epoch in range(epochs):
        for TCR,Vs,Js,TCR_len ,alpha, aVs,aJs,alpha_len,MHC_As,MHC_class, epitope,epi_len,label,weight in dataset:
            optimizer.zero_grad()
            pred = model(TCR,epitope,alpha,Vs,Js,TCR_len,aVs,aJs,alpha_len,MHC_As,MHC_class,epi_len)
            
            
            loss = F.binary_cross_entropy(pred.view(-1),label.to(torch.float),weight= pl_model.weight_fun(label,weight))
            loss.backward()
            optimizer.step()
            # print(scheduler.get_last_lr())
            if scheduler.get_last_lr()[0] == lr_min:
                swa_model.update_parameters(model)
                # print("update")
            scheduler.step()
    print("n_averaged ", swa_model.n_averaged)
            


    # transferring avg model weights 

    for w_dest, w_swa_model in zip(model.parameters(), swa_model.parameters()):
            device = w_dest.device
            w_swa_ = w_swa_model.detach().to(device)
            
            w_dest.detach().copy_(w_swa_)
    

    file =str(pl_model.hparams["hparams"]['version'])+"_manual_swa"+ str(lr_max) +"_" +str(cycle_len)+ "_" +str(epochs)+"_"

    midx = 0
    for filename in os.listdir(dir_path):
        if (file) in filename:
            try:
                swa_idx = int(filename[-6:-4])
            except:
                swa_idx = 0
            midx = max(midx,swa_idx)
    midx += 1
    nbr = str(midx) if midx >9 else ("0" +str(midx) )
    path = dir_path + file + nbr
    # torch.save(model,path +".pth")
    # print("manual SWA saved")
    if ret_path_name:
        return path


def do_SWA_const_lr(pl_model:LitEPICTRACE,lr,weight_save_freq=1,on_epoch=True,epochs=10,ret_path_name=True,save_dir=None,tlogger=None):
    print("manual SWA with params ",lr," ",epochs)

    if save_dir:
        dir_path = save_dir
    else:
        dir_path =  os.getcwd() +'/loggingDir22/folder22/version_'+ str(pl_model.hparams["hparams"]['version']) +'/checkpoints/' 
        
        if not os.path.exists(dir_path):
            dir_path = os.getcwd() +'/loggingDir22/folder22/versions'+ str(pl_model.hparams["hparams"]['version'])[:3]+'_' + str(pl_model.hparams["hparams"]['version'])[-2] +  '/version_'+ str(pl_model.hparams["hparams"]['version']) +'/checkpoints/'
    assert os.path.exists(dir_path)

    model = pl_model.EPICTRACE_model
    model.train()
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    # pl_model.configure_optimizers()["optimizer"]
   
    dataset = pl_model.train_dataloader()
    ite = 0
    for epoch in range(epochs):
        for TCR,Vs,Js,TCR_len ,alpha, aVs,aJs,alpha_len,MHC_As,MHC_class, epitope,epi_len,label,weight in dataset:
            optimizer.zero_grad()
            pred = model(TCR,epitope,alpha,Vs,Js,TCR_len,aVs,aJs,alpha_len,MHC_As,MHC_class,epi_len)
            
            
            loss = F.binary_cross_entropy(pred.view(-1),label.to(torch.float),weight= pl_model.weight_fun(label,weight))
            
            loss.backward()
            optimizer.step()
            pl_model.logger.log_metrics({"SWA_train_loss":loss.item()})
            # tlogger.log_metrics({"SWAt_train_loss":loss.item()})
            if not on_epoch and ite % weight_save_freq==0:
                swa_model.update_parameters(model)
            ite +=1
            # print(scheduler.get_last_lr())
        if on_epoch and epoch % weight_save_freq==0:
                swa_model.update_parameters(model)
    print("n_averaged ", swa_model.n_averaged)
            


    # transferring avg model weights 

    for w_dest, w_swa_model in zip(model.parameters(), swa_model.parameters()):
            device = w_dest.device
            w_swa_ = w_swa_model.detach().to(device)
            
            w_dest.detach().copy_(w_swa_)
    

    file =str(pl_model.hparams["hparams"]['version'])+"_manual_swa_c"+ str(lr) +"_" +str(weight_save_freq)+ "_" +str(epochs)+"_"

    midx = 0
    for filename in os.listdir(dir_path):
        if (file) in filename:
            try:
                swa_idx = int(filename[-6:-4])
            except:
                swa_idx = 0
            midx = max(midx,swa_idx)
    midx += 1
    nbr = str(midx) if midx >9 else ("0" +str(midx) )
    path = dir_path + file + nbr
    # torch.save(model,path +".pth")
    # print("manual SWA saved")
    if ret_path_name:
        return path