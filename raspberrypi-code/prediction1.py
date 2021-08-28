import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import librosa
import numpy as np
import os
import time

def load_model(epoch, model, path='./'):
    
    # file name and path 
    filename = path + 'neural_network_{}.pt'.format(epoch)
    
    # load the model parameters 
    model.load_state_dict(torch.load(filename))
    
    
    return model

class CNN_Baby(nn.Module):
    def __init__(self):
        super(CNN_Baby,self).__init__()
        # self.cv1=nn.Conv2d(1,16,5,stride=[1],padding=[0],dilation=[1])
        # self.cv2=nn.Conv2d(16,32,5,stride=[1],padding=[0],dilation=[1])
        # #self.fc=nn.Linear(322875,4)
        # self.fc=nn.Linear(12800,10)
        self.fc1=nn.Linear(1025*315,4)
        # self.fc2=nn.Linear(500,100)
        self.fc3=nn.Linear(500,4)
        #
    def forward(self,x):
        # out=F.relu(self.cv1(x))
        # out=F.relu(self.cv2(out))
        # out=out.view(BATCH_SIZE,-1)
        # out=self.fc(out)
        out=self.fc1(x)
        # out=F.relu(self.fc2(out))
        #out=F.relu(self.fc3(out))
        return out


path = './'
epoch=20
cnn=CNN_Baby()

load_model(epoch,cnn,path)


def readFile(filepath):
    y,sr=librosa.load(filepath)

    D=librosa.stft(y)

    D_real, D_imag = np.real(D), np.imag(D)
    #print(D_imag)
    #D_energy = np.real(D)
    a=D_real**2+D_imag**2
    #print(a)
    D_energy = np.sqrt(D_real**2+D_imag**2)
    # a=D_real**2+D_imag**2
    # if a>=0:
    #     D_energy = np.sqrt(D_real**2+D_imag**2)
    # else:
    #     print(a)
    #     D_energy=0
    
    #result=np.log(D_energy)
    norm = librosa.util.normalize(D_energy)
    #display.specshow(norm, y_axis='log', x_axis='time')
    #plt.imshow(result)
    #plt.savefig("1.png")

    #plt.plot(result)
    #plt.show()
    result=np.pad(norm,([(0,0),(0,315-len(norm[0]))]),'constant')
    return result

def get_class( thelist ):
    max=thelist[0]
    index=0
    i=0
    for element in thelist:
        if(max<element):
            max=element
            index=i
        i=i+1;
    #print (index)
    if(index==0):
        return 'silence'
    elif(index==1):
        return 'noise'
    elif(index==2):
        return 'Laugh'
    elif(index==3):
        return 'crying'

def get_label_index(tensor_pred):
    npray = tensor_pred.detach().numpy()
    index = [npray[0]]
    for x in range(0,len(npray) -1):
        npray [x+1] > npray[x]
        index = (x+1)
    return index


#read a data from audio mic
os.system('arecord -D plughw:1,0 -d 6 -f S16_LE -c1 -r44100 -t wav signal_6s.wav')
time.sleep(2)
f=readFile("signal_6s.wav")
p=cnn(torch.tensor(f).view(-1,1025*315))
print(" file test:{}".format(p))
print ("precicted class  :"+get_class(get_label_index(p)[0]))


           


