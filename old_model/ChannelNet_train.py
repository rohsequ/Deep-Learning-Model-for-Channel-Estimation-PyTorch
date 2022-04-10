import numpy as np
import math
from models import interpolation , SRCNN_train , SRCNN_model, SRCNN_predict , DNCNN_train , DNCNN_model , DNCNN_predict
#from scipy.misc import imresize
from scipy.io import loadmat
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # load datasets 
    channel_model = "VehA"
    SNR = 22
    Number_of_pilots = 48
    import pdb;pdb.set_trace()
    perfect = loadmat("Perfect_"+ channel_model+".mat") ['My_perfect_H']
    noisy_input = loadmat("Noisy_" + channel_model + "_" + "SNR_" + str(SNR) + ".mat") [channel_model+"_noisy_"+ str(SNR)]             
                      
    interp_noisy = interpolation(noisy_input , SNR , Number_of_pilots , 'rbf')

    perfect_image = np.zeros((len(perfect),72,14,2))
    perfect_image[:,:,:,0] = np.real(perfect)
    perfect_image[:,:,:,1] = np.imag(perfect)
    perfect_image = np.concatenate((perfect_image[:,:,:,0], perfect_image[:,:,:,1]), axis=0).reshape(2*len(perfect), 72, 14, 1)
    
    
    ####### ------ training SRCNN ------ #######
    idx_random = np.random.rand(len(perfect_image)) < (1/9)  # uses 32000 from 36000 as training and the rest as validation
    train_data, train_label = interp_noisy[idx_random,:,:,:] , perfect_image[idx_random,:,:,:]
    val_data, val_label = interp_noisy[~idx_random,:,:,:] , perfect_image[~idx_random,:,:,:]    
    SRCNN_train(train_data ,train_label, val_data , val_label , channel_model , Number_of_pilots , SNR )
    
   
    ####### ------ prediction using SRCNN ------ #######
    srcnn_pred_train = SRCNN_predict(train_data, channel_model , Number_of_pilots , SNR)
    srcnn_pred_validation = SRCNN_predict(train_data, channel_model , Number_of_pilots , SNR)  
                      
                      
    ####### ------ training DNCNN ------ #######
    # DNCNN_train(input_data, channel_model , num_pilots , SNR):
                      

    
