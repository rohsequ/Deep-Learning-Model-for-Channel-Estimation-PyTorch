import numpy as np
import math
#from scipy.misc import imresize
import scipy.io
import matplotlib.pyplot as plt
from scipy import interpolate
from pytorch_models import train_SRCNN, train_DnCNN
import torch
from torch.utils.data import TensorDataset
from datetime import datetime



##################################################################
#
##################################################################

n_epochs = 5
load_from_checkpoint = True


def psnr(target, ref):
    # assume RGB image
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)

def interpolation(noisy , SNR , Number_of_pilot , interp):
    N, N_S, N_D = noisy.shape
    noisy_image = np.zeros((N, N_S, N_D, 2))

    noisy_image[:,:,:,0] = np.real(noisy)
    noisy_image[:,:,:,1] = np.imag(noisy)


    if (Number_of_pilot == 48):
        idx = [14*i for i in range(1, 72,6)]+[4+14*(i) for i in range(4, 72,6)]+[7+14*(i) for i in range(1, 72,6)]+[11+14*(i) for i in range(4, 72,6)]
    elif (Number_of_pilot == 16):
        idx= [4+14*(i) for i in range(1, 72,9)]+[9+14*(i) for i in range(4, 72,9)]
    elif (Number_of_pilot == 24):
        idx = [14*i for i in range(1,72,9)]+ [6+14*i for i in range(4,72,9)]+ [11+14*i for i in range(1,72,9)]
    elif (Number_of_pilot == 8):
      idx = [4+14*(i) for  i in range(5,72,18)]+[9+14*(i) for i in range(8,72,18)]
    elif (Number_of_pilot == 36):
      idx = [14*(i) for  i in range(1,72,6)]+[6+14*(i) for i in range(4,72,6)] + [11+14*i for i in range(1,72,6)]

    r = [x//14 for x in idx]
    c = [x%14 for x in idx]

    interp_noisy = np.zeros((N, N_S, N_D, 2))

    for i in range(len(noisy)):
        z = [noisy_image[i,j,k,0] for j,k in zip(r,c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,:,:,0] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,0] = z_intp
        z = [noisy_image[i,j,k,1] for j,k in zip(r,c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,:,:,1] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,1] = z_intp


    # interp_noisy = np.concatenate((interp_noisy[:,:,:,0], interp_noisy[:,:,:,1]), axis=0)
   
    
    return interp_noisy


if __name__ == "__main__":
    # load datasets 
    channel_model = "VehA"
    SNR = 12
    Number_of_pilots = 48
    
    print("Reading the Noisy and Perfect Data files in .mat format.")

    perfect = scipy.io.loadmat("data\Perfect_H_40000.mat") ['My_perfect_H']
    noisy_input = scipy.io.loadmat("data\My_Noisy_H_12.mat") ['My_noisy_H']  

    print(" \n ")
    print("Read the Noisy and Perfect Data files.")  
    print(" \n ")   
    print("Interpolating the noisy data into LR images")
    print(f"SNR: {SNR}\nNumber of pilots: {Number_of_pilots}")
    interp_noisy = interpolation(noisy_input , SNR , Number_of_pilots , 'rbf')

    n, N_s, N_d = perfect.shape
    perfect_image = np.zeros((n, N_s, N_d, 2))
    perfect_image[:,:,:,0] = np.real(perfect)
    perfect_image[:,:,:,1] = np.imag(perfect)
    # perfect_image = np.concatenate((perfect_image[:,:,:,0], perfect_image[:,:,:,1]))

    print("==========================================================")
 
    ####### ------ training SRCNN ------ #######
    interp_noisy = interp_noisy.transpose((0,3,1,2))
    perfect_image = perfect_image.transpose((0,3,1,2))
    
    t_interp_noisy = torch.Tensor(interp_noisy)
    t_perfect_image = torch.Tensor(perfect_image)
    my_dataset = TensorDataset(t_interp_noisy, t_perfect_image) 

    train_size = int(0.8 * len(my_dataset))
    test_size = len(my_dataset) - train_size
     


    device=torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model_name = "SRCNN"

    print("Data is ready for training.\nTraining the SRCNN first....")

    train_SRCNN(train_size, test_size, dataset=my_dataset, n_epochs=n_epochs, model_name=model_name, device=device, load_from_checkpoint=load_from_checkpoint)

    print("Data is ready for training.\nTraining the DnCNN first....")
    model_name = "DnCNN"

    train_DnCNN(train_size, test_size, dataset=my_dataset, n_epochs=n_epochs, model_name=model_name, device=device, load_from_checkpoint=load_from_checkpoint, path_to_SRCNN="saved_models\SRCNN_checkpoint_latest.pt")


    print("Trained weights stored in 'saved_models' folder.")



