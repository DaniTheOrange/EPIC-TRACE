from logging import Logger
import pytorch_lightning as pl
import pickle
from torch._C import Argument

import os
from pytorch_lightning.callbacks import EarlyStopping ,ModelCheckpoint , StochasticWeightAveraging ,LearningRateMonitor
import argparse
import SWA
from EPICTRACE_model import LitEPICTRACE, EPICTRACE


parser = argparse.ArgumentParser()
parser.add_argument("model",type=str,help="Dummy argument to be compatible with saved models")
parser.add_argument("-d","--dataset", type=str, default='IEVDJcor310fPEpivalR_17_60tpp3_CV/ab_b0',help="Name of dataset to train and validate the model with. Located in the 'data/' folder. Train and validation files ends with '_train_data' and '_validate_data' + extension= '.gz' / '.csv'")
parser.add_argument("--lr",type=float,default=0.001,help="Learning rate to be used.")
parser.add_argument("--batch_size",type=int,default=128)
parser.add_argument("-c","--cpus",type=int,help="The number of CPUs to use for training. If not specified, all available CPUs will be used.")
parser.add_argument("-i","--input_embedding_data",type=str,help="Name of datafile with pickled embedding dict. Located in the data folder.")
parser.add_argument("-l","--input_embedding_dim",type=int,default=1045)
parser.add_argument("-T","--TCR_max_length",type=int,default=25)
parser.add_argument("-E","--epitope_max_length",type=int,default=16)
parser.add_argument("--filter_sizes", nargs="+",type=int, default=[7],help="Filter sizes for the convolutional layers.")
parser.add_argument("--epitope_filter_sizes", nargs="+",type=int,help="Filter sizes for the epitope convolutional layers. Defaults to the same as filter_sizes.")
parser.add_argument("--filter_nums",nargs="+",type=int,default=[100],help="Number of filters for the convolutional layers.")
parser.add_argument("-f","--conv_2_filter_size",type=int,default=0,help="Filter size for the second convolutional layer. If 0, no second convolutional layer will be used.")
parser.add_argument("--num_tf_blocks",type=int,default=0,help="Number of transformer blocks to use. If 0, no transformer blocks will be used only Self-Attention without the feed forward parts.")
parser.add_argument("--num_heads",type=int,default =5,help="Number of heads for the self attention layers.")
parser.add_argument("--skip",default=False,action="store_true",	help="Use skip connections from input to chain concatenation after transformer/self-attention.")
parser.add_argument("-g","--gpus",type=int,default=1,help="The number of GPUs to use for training.")
parser.add_argument("--max_epochs",type=int,help="The maximum number of epochs to train for.")
parser.add_argument("-v","--version", type=int,	help="The version name of the model. Used for logging and saving checkpoints.")
parser.add_argument("-p","--pos_encoding",default=False,action="store_true",help="Use positional encoding for the transformer blocks.")
parser.add_argument("--BN",default=False,action="store_true",help="Use Batch Norm after first convolution(s).")
parser.add_argument("--dropout",type=float,default =0.2,help="Dropout rate for dropout layers after conv.")
parser.add_argument("--dropout_attn",type=float,default =0.2,help="Dropout rate for dropout layers in transformer blocks or self attention.")
parser.add_argument("--dropout2",type=float,default =0.45,help="Dropout rate for dropout layers after linear.")
parser.add_argument("--load",default=False,action="store_true",help="Load a pretrained model specified by --version.")
parser.add_argument("--collate",type=str,default="embedding_one_hot", help="The collate function to use for the dataset.")
parser.add_argument("--output_mhc",action="store_true", help="Use MHC information for prediction.")
parser.add_argument("--output_vj",action="store_true",help="Use VJ information for prediction.")
parser.add_argument("--output_seq_len",action="store_true", help="Dummy argument to be compatible with saved models") 
parser.add_argument("--only_beta",default=False , action="store_true",help="Only use beta chain for prediction.")
parser.add_argument("--only_alpha",default=False , action="store_true",help="Only use alpha chain for prediction.")
parser.add_argument("--weight_fun",type=str,help="The weighting function to use for the loss Defaults to 'Label*(4.0) + 1', If --weight_fun == 'Linear' the weight specified in the Weight column of the train data file is used.")
parser.add_argument("--mhc_hi",action="store_true",help="Use MHC information in a hierarcical way for prediction.")
parser.add_argument("--wd",type=float,default=1e-2,help="Weight decay for the optimizer.")
parser.add_argument("--test_task",type=int,default= -1,help="The task index i.e. 1 TPP2 and 2 for TPP3 for automatic (micro) AUROC and AP testing for file '<--dataset>_tpp<2/3>_data' + extension= '.gz' / '.csv'. If -1, no automatic testing will be done.")
parser.add_argument("--lr_s",default=False,action="store_true", help="Use a learning rate scheduler, benficial for TPP3.")
parser.add_argument("--manual_SWA",action="store_true",default=False, help="Use Stochastic weight averaging (recommended)")
parser.add_argument("--SWA_max_lr",type=float,default=0.0005,help="The maximum learning rate to use for the SWA optimizer.")
parser.add_argument("--SWA_epochs",type=int,default=10,help="The number of epochs to train for with the SWA optimizer after initial training.")
parser.add_argument("--SWA_cycle",type=int,default=50, help="The cycle length for SWA (when the model params are saved to be used in the average)")
parser.add_argument("--SWA_on_epoch",action="store_true",default=False,help="Use SWA on epoch instead of iteration.")
parser.add_argument("--SWA_const",action="store_true",default=False,help="Use constant SWA instead of cyclic SWA (recommended).")
parser.add_argument("--add_01",action="store_true",default=False,help="assume 01 allele if only V or J gene is specified and try creating the Long TCR sequence given 01 allele")
parser.add_argument("--feed_forward",action="store_true",default=False,help="Use feed forward in addition to convolution from input embeddings")
parser.add_argument("--lr_find",action="store_true",default=False,help="Use pytorch lighting learning rate finder to find optimal learning rate")
parser.add_argument("--es_epoch_3",action="store_true",default=False,help="Use early stopping after 3 epochs instead of 8 iterations")
parser.add_argument("--only_CDR3",action="store_true",default=False,help="Only use CDR3 sequence for prediction, affects the trainloader, this can be manually made by swithcing the Long columns to have the CDR3 sequences instead of the full TCR sequences")
parser.add_argument("--load_MHC_dicts",action="store_true",default=False,help="Load MHC dictionaries from file 'data/MHC_all_dict.bin' or files 'data/MHC_lvl2nd_dict.bin' and 'data/MHC_lvl3rd_dict.bin' for hierarcical MHC instead of the hard coded values.")

hparams = parser.parse_args()
model =LitEPICTRACE(vars(hparams),EPICTRACE)


print("version: ",hparams.version)
print(model)


checkpoint_callback = ModelCheckpoint( monitor='valid_ap' ,mode='max') # ,save_last=True save space
if hparams.es_epoch_3:
	early_stopping = EarlyStopping('valid_ap',mode='max',patience =3,check_on_train_epoch_end=True)
else:
	early_stopping = EarlyStopping('valid_ap',mode='max',patience =8)
print("valid_ap")
swa = StochasticWeightAveraging(swa_lrs=0.0003,swa_epoch_start=45)

lr_monitor = LearningRateMonitor(logging_interval='step')

logger=pl.loggers.TensorBoardLogger("loggingDir22",name="folder22",version=hparams.version)
trainer = pl.Trainer(gpus=hparams.gpus,logger=logger,max_epochs=hparams.max_epochs,callbacks=[early_stopping,checkpoint_callback,lr_monitor],progress_bar_refresh_rate=0,auto_lr_find=hparams.lr_find) # early_stopping removed


if hparams.load:
	checkpoint_dir = os.getcwd() +'/loggingDir22/folder22/version_'+ str(hparams.version) +'/checkpoints/'
	if not os.path.exists(checkpoint_dir):
		checkpoint_dir = os.getcwd() +'/loggingDir22/folder22/versions'+ str(hparams.version)[:3]+'_' + str(hparams.version)[-2] +  '/version_'+ str(hparams.version) +'/checkpoints/'
		assert os.path.exists(checkpoint_dir)
	idx=0
	e=0
	for i,strr in enumerate(os.listdir(checkpoint_dir)):
		start = strr.find("epoch=")
		

		if start != -1:
			ee=int(strr[start+6:strr.find("-")])
			if ee>=e:
				e=ee
				idx=i
	path = checkpoint_dir+ os.listdir(checkpoint_dir)[idx]
	print(path)
	model = LitEPICTRACE.load_from_checkpoint(path)
	
vv=trainer.validate(model)
print("vv",vv)

if hparams.max_epochs>0:
	trainer.fit(model)
	path=checkpoint_callback.best_model_path
	print(path)




model = LitEPICTRACE.load_from_checkpoint(path)
model.test_task = hparams.test_task

trainer.validate(model)
if hparams.test_task != -1:
	ret =trainer.test(model)[0]
	if hparams.max_epochs>0:
		with open("results/result_dicts/"+ str(hparams.version)+ ".bin",'wb') as handle:
			pickle.dump(ret,handle)

if hparams.manual_SWA:
	checkpoint_dir = os.getcwd() +'/loggingDir22/folder22/version_'+ str(hparams.version) +'/checkpoints/'
	if not os.path.exists(checkpoint_dir):
		checkpoint_dir = os.getcwd() +'/loggingDir22/folder22/versions'+ str(hparams.version)[:3]+'_' + str(hparams.version)[-2] +  '/version_'+ str(hparams.version) +'/checkpoints/'
		assert os.path.exists(checkpoint_dir)
	if hparams.SWA_const:
		path = SWA.do_SWA_const_lr(model,hparams.SWA_max_lr,hparams.SWA_cycle,hparams.SWA_on_epoch,hparams.SWA_epochs)#,tlogger=trainer.logger)
	else:
		path = SWA.do_SWA(model,hparams.SWA_max_lr,0.00001,hparams.SWA_cycle,hparams.SWA_epochs,save_dir=checkpoint_dir)
	if path:
		trainer.save_checkpoint( path +".ckpt")
	else:
		trainer.save_checkpoint( os.getcwd() +'/loggingDir22/folder22/version_'+ str(hparams.version) +'/checkpoints/' +"manual_SWA.ckpt")
		print("PROBLEM saving")
	
	trainer.validate(model)
	if hparams.test_task != -1:
		ret =trainer.test(model)[0]
	
		with open("results/result_dicts/"+ str(hparams.version) +"_" + str(hparams.SWA_max_lr) +"_" +str(hparams.SWA_cycle)+ "_" +str(hparams.SWA_epochs)+ ".bin",'wb') as handle:
			pickle.dump(ret,handle)



