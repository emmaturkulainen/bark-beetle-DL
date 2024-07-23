"""
2021-2022

@author: KK
edited by: ET
"""


import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import reshape


#inspired by Ji, M.; Liu, L.; Du, R.; Buchroithner, M.F. 
#A Comparative Study of Texture and Convolutional Neural Network Features for Detecting Collapsed Buildings After Earthquakes Using Pre- and Post-Event Satellite Imagery. 
#Remote Sens. 2019, 11, 1202. https://doi.org/10.3390/rs11101202
 

# Kirsi's 2D-CNN
class CNN2D(nn.Module):
    def __init__(self, in_channels, params={"dropout": 0.5}, classes=4):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2,2), 
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)))
            
        self.layer2 = nn.Sequential(
            nn.Linear(6*6*64, 64),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(64, classes))
 
    def forward(self, x):
        out = self.layer1(x)
        out = torch.flatten(out, 1) 
        out = self.layer2(out)
        return out   




# Kirsi's 3D-CNN 
class CNN1_3D(nn.Module):
    def __init__(self, in_channels, params={"dropout":0.5}, classes=4):
        super().__init__()
        self.in_channels = in_channels
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,3,3), stride=1, padding='same'),
            #nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding='same'),
            #nn.BatchNorm3d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool3d((2,2,2)), 
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding='same'),
            #nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(6))
            
        self.layer2 = nn.Sequential(
            nn.Linear(6*6*6*64, 64), 
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(64, classes))
 
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.layer1(x)
        out = torch.flatten(out, 1) 
        out = self.layer2(out)
        return out


'''Nezami et al., 2020, https://doi.org/10.3390/rs12071070'''
class CNN2_3D(nn.Module):
    def __init__(self, in_channels, params={"dropout":0.5}, classes=4):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 20, kernel_size=(5,5,5), stride=1, padding=0),
            #nn.BatchNorm3d(20),
            nn.MaxPool3d((3, 3, 3)),
            nn.Conv3d(20, 50, kernel_size=(5,5,5), stride=1, padding=0),
            #nn.BatchNorm3d(50),
            nn.MaxPool3d((3,3,3)), 
            nn.ReLU(inplace=True),
            nn.Conv3d(50, 3, kernel_size=(1,1,1), stride=1, padding=0)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(3*3*3*3, 128),
            nn.Linear(128, classes)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.layer1(x)
        out = torch.flatten(out, 1) 
        out = self.layer2(out)
        return out

'''Pi et al., 2021, https://doi.org/10.1016/j.ecoinf.2021.101278'''
class CNN3_3D(nn.Module):
    def __init__(self, in_channels, params={"dropout":0.5}, classes=4):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 5, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool3d((1,1,1)),
            nn.Conv3d(5, 10, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool3d((1,1,1)), 
            nn.Conv3d(10, 15, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool3d((3,1,1)))

        self.layer2 = nn.Sequential(
            nn.Linear(15*44*44*13, 1024),
            nn.Linear(1024,classes)
            #nn.Softmax()
        )
 
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.layer1(x)
        out = torch.flatten(out, 1) 
        out = self.layer2(out)
        return out



'''Zhang et al., 2020, https://doi.org/10.1016/j.rse.2020.111938'''
class CNN4_3D(nn.Module):
    def __init__(self, in_channels, params={"dropout":0.5}, classes=4):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(3,3,7), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv3d(4, 8, kernel_size=(3,3,7), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=(3,3,7), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3,3,7), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,3,7), stride=1, padding='same'),
            #nn.ReLU(),
            #nn.Conv3d(64, 128, kernel_size=(3,3,7), stride=1, padding='same'),
            nn.MaxPool3d((2,1,1)),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Dropout(params["dropout"]), 
            nn.Linear(64*50*50*23, 64),
            nn.Dropout(params["dropout"]),
            nn.Linear(64, classes)
        )
 
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.layer1(x)
        out = torch.flatten(out, 1) 
        out = self.layer2(out)
        return out

'''Yu et al, 2020, https://doi.org/10.1109/JSTARS.2020.2983224'''
class CNN1_2D3D(nn.Module):
    def __init__(self, in_channels, params={"dropout":0.5}, classes=4):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 200, kernel_size=(3,3), stride=1, padding='same')
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(200, 1, kernel_size=(3,3,3), stride=1, padding='same'),
            nn.MaxPool3d((1,3,3)),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(1, 200, kernel_size=(3,3), stride=1, padding=1))

        self.layer4 = nn.Sequential( 
            nn.Linear(16*16*200, 250),
            nn.Linear(250,classes)
        )
 
    def forward(self, x):
        out = self.layer1(x)
        out = torch.unsqueeze(out, 2)
        out = self.layer2(out)
        out = torch.flatten(out, 1,2) 
        out = self.layer3(out)
        out = torch.flatten(out, 1) 
        out = self.layer4(out)
        return out

'''Ge et al., 2020, https://doi.org/10.1109/JSTARS.2020.3024841'''
class CNN2_2D3D(nn.Module):
    def __init__(self, in_channels, params={"dropout":0.5}, classes=4):
        super().__init__()

        self.branch1_layer1 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(7, 7, 7), stride=1, padding=0),
            nn.ReLU()
        )

        self.branch1_layer2 = nn.Conv2d(6*40, 64, kernel_size=(3,3), stride=1, padding=0)

        self.branch2_layer1 = nn.Sequential( 
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5), stride=1, padding=0),
            nn.Conv3d(6, 12, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.ReLU()
        )
        
        self.branch2_layer2 = nn.Conv2d(12*40, 64, kernel_size=3, stride=1, padding=0)

        self.branch3_layer1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.Conv3d(8, 12, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.Conv3d(12, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.ReLU()
        )
        
        self.branch3_layer2 = nn.Conv2d(32*40, 64, kernel_size=3, stride=1, padding=0)
        
        self.fc = nn.Sequential(
            nn.Linear(42*42*64*3, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16), 
            nn.Linear(16, classes)
        )


    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        b1l1 = self.branch1_layer1(x)
        b1l1 = torch.flatten(b1l1, 1, 2)
        b1 = self.branch1_layer2(b1l1)
        b2l1 = self.branch2_layer1(x)
        b2l1 = torch.flatten(b2l1,1,2)
        b2 = self.branch2_layer2(b2l1)
        b3l1 = self.branch3_layer1(x)
        b3l1 = torch.flatten(b3l1, 1,2)
        b3 = self.branch3_layer2(b3l1)
        con1 = torch.cat((b1,b2),1)
        con2 = torch.cat((con1, b3), 1)
        out = torch.flatten(con2, 1)
        out = self.fc(out)
        return out



class Hyper3DNet(nn.Module):
    def __init__(self, in_channels, params={"dropout":0.5}, classes=4): 
        super().__init__()
        self.classes = classes  

        self.feature_extractor1 = nn.Sequential(
            nn.Conv3d(1, out_channels=8, kernel_size=(3,3,5), padding='same'),
            nn.BatchNorm3d(8), 
            nn.ReLU()
        )
        self.feature_extractor2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3,3,5), padding='same'),
            nn.BatchNorm3d(8), 
            nn.ReLU()
        )
        self.feature_extractor3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=8, kernel_size=(3,3,5), padding='same'), 
            nn.BatchNorm3d(8), 
            nn.ReLU()
        )
        self.feature_extractor4 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=8, kernel_size=(3,3,5), padding='same'), 
            nn.BatchNorm3d(8), 
            nn.ReLU())                                      
        
        self.spatial_encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=32*in_channels, out_channels=32, kernel_size=3, stride= 1, padding='same', groups=32), 
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding='same'), 
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.spatial_encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride = 1, padding='same', groups=16),  #padding='same' does not support stride!=1
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1,  padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
                nn.Linear(50*50*16, 256),
                nn.ReLU(),
                nn.Dropout(params["dropout"]),
                nn.Linear(256, classes))
     
    def forward(self, x):
        #print("x", x.size())
        x = torch.unsqueeze(x, 1)
        t1 = self.feature_extractor1(x)
        t2 = self.feature_extractor2(t1)
        #print("t2", t2.size())
        con1 = torch.cat((t1, t2), 1) 
        #print("con1", con1.size())
        t3 = self.feature_extractor3(con1)
        con2 = torch.cat((t3, con1), 1)
        t4 = self.feature_extractor4(con2)
        con3 = torch.cat((t4, con2), 1) 
        #print("con3", con3.size())
        #3D to 2D
        #re1 = reshape(con3, (con3.shape[0], self.img_shape[1] * 32, self.img_shape[2], self.img_shape[3])) 
        re1 = torch.flatten(con3, 1,2)
        #re1 = reshape(con3, (con3.shape[1], con3.shape[2],con3.shape[3] * con3.shape[4]))
        #print("re1", re1.size())
        
        s1 = self.spatial_encoder1(re1)
        #print("s1", s1.size())
        s2 = self.spatial_encoder2(s1)
        #print("s2", s2.size())        
        s3 = self.spatial_encoder2(s2)
        s4 = self.spatial_encoder2(s3)
        #print("s4", s4.size())
        f1 = torch.flatten(s4, 1) 
        #print("f1", f1.size())
        out = self.fc_layers(f1)
        return out   


'''
#ILKKA PÖLÖNEN Keras implementation
def Hyper3DNet():
 
    def featurextractor(cube,nro):
        
        tensor = Conv3D(nro,kernel_size=(3,3,5),strides=(1,1,1),padding='same')(cube)
        tensor = BatchNormalization()(tensor)
        tensor = ReLU()(tensor) 
        
        return tensor
    
    def spatialencoder(cube,strid):
        
        tensor = SeparableConv2D(16,strides=strid,kernel_size=(3,3),padding='same',dilation_rate=1)(cube)
        tensor = BatchNormalization()(tensor)
        tensor = ReLU()(tensor) 
        
        return tensor
    
    cube = Input(shape=(25,25,55,1))
    
    t1 = featurextractor(cube,8)
    t2 = featurextractor(t1,8)
    con1 = concatenate([t1, t2],axis=4)
    t3 = featurextractor(con1,8)
    con2 = concatenate([t3, con1],axis=4)
    t4 = featurextractor(con2,8)
    con3 = concatenate([t4, con2],axis=4)
    
    re1 = Reshape((con3.shape[1], con3.shape[2],con3.shape[3] * con3.shape[4]))(con3)
    
    s1 = spatialencoder(re1,1)
    s2 = spatialencoder(s1,2)
    s3 = spatialencoder(s2,2)
    s4 = spatialencoder(s3,2)
    
    f1 = Flatten()(s4)

    fc1 = Dense(256, activation='relu')(f1)
    fc1 = Dropout(0.5)(fc1)

    output = Dense(3, activation='softmax')(fc1)
    
    model = Model(inputs=cube, outputs=output)
    
    return model

# Morales et al. "Reduced-cost hyperspectral convolutional neural networks" https://www.cs.montana.edu/sheppard/pubs/jars-2020.pdf'''

 

