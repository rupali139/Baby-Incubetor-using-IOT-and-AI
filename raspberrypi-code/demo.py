
import sys
import urllib
from time import sleep
import Adafruit_DHT as dht
import numpy as np
from pulsesensor import Pulsesensor
import time
import RPi.GPIO as GPIO
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import os


# Enter Your API key here
myAPI = 'Thingspeak-API-key' 
# URL where we will send the data, Don't change it
baseURL = 'https://api.thingspeak.com/update?api_key=%s' % myAPI 


channel = 2

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(channel, GPIO.OUT)
##################################################################################################
def DHT22_data():
	# Reading from DHT22 and storing the temperature and humidity
	humi, temp = dht.read_retry(dht.DHT22, 4) 
	return humi, temp

def buzzer_off(pin):
    GPIO.output(pin, GPIO.LOW)  # Turn motor on


def buzzer_on(pin):
    GPIO.output(pin, GPIO.HIGH)  # Turn motor off

####################################################################################################
def load_model(epoch, model, path='./'):
    
    # file name and path 
    filename = path + 'neural_network_{}.pt'.format(epoch)
    
    # load the model parameters 
    model.load_state_dict(torch.load(filename))
    
    
    return model

def readFile(filepath):
    y,sr=librosa.load(filepath)

    D=librosa.stft(y)

    D_real, D_imag = np.real(D), np.imag(D)
    a=D_real**2+D_imag**2
    #print(a)
    D_energy = np.sqrt(D_real**2+D_imag**2)
    norm = librosa.util.normalize(D_energy)
    #display.specshow(norm, y_axis='log', x_axis='time')
    #plt.imshow(result)
    plt.savefig("1.png")

    #plt.plot(result)
    plt.show()
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
    
    return index
            
            

def get_label_index(tensor_pred):
    npray = tensor_pred.detach().numpy()
    index = [npray[0]]
    for x in range(0,len(npray) -1):
        npray [x+1] > npray[x]
        index = (x+1)
    return index

########################################################################################################
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

#############################################################################################################
p = Pulsesensor()
p.startAsyncBPM()
path = './'
epoch=20
cnn=CNN_Baby()
load_model(epoch,cnn,path)

while True:
    try:
            bpm = p.BPM
            os.system("arecord -D plughw:1,0 -d 6 -f S16_LE -c1 -r44100 -t wav signal_6s.wav")
            sleep(3)
            f=readFile("signal_6s.wav")
            p=cnn(torch.tensor(f).view(-1,1025*315))
            print("Crying file test:{}".format(p))
            cryPrediction=get_class(get_label_index(p)[0])
            if bpm > 0:
                print("BPM: %d" % bpm)
                humi, temp = DHT22_data()
                if isinstance(humi, float) and isinstance(temp, float):
                # Formatting to two decimal places
                    humi = '%.2f' % humi 
                    temp = '%.2f' % temp
                    print (humi)
                    print (temp) 
              
              # If Reading is valid

              # Sending the data to thingspeak
                conn = urllib.urlopen(baseURL + '&field1=%s&field2=%s&field3=%s&field4=%s' % (temp, humi,bpm,cryPrediction))
                #print conn.read()
              # Closing the connection
                conn.close()

                if bpm < 70:
                    buzzer_on(channel)
                else:
                    buzzer_off(channel)

            else:
                print("No Heartbeat found")
                time.sleep(2)

            sleep(5)
    except:
        GPIO.cleanup()
        p.stopAsyncBPM()
        break









