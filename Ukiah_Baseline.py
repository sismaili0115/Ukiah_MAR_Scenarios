# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:42:31 2019


"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:29:50 2019


# -*- coding: utf-8 -*-
"""

import flopy
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
import csv

os.getcwd()

##### Change the folderpath
os.chdir ('D:\\A_Ukiah_SA\\Local_SA')#os.getcwd()
#dname=os.path.dirname(os.path.abspath(__file__))
#os.chdir (dname)
#print (os.getcwd())
#%% Model name
##### Assign the model name and location
model_name = 'Flopy_Outputs_SA\\Ukiah_0.2'

# Assign the MODFLOW executable location
mf = flopy.modflow.Modflow(model_name, exe_name=  'D:\\WRDAPP\\MF2005.1_12\\bin\\mf2005.exe')

#%% 1. Model discretisation :dis = flopy.modflow.ModflowDis(mf, nlay=self.nlay, nrow=self.nrow, ncol=self.ncol, nper=nper,delr=self.cellsize,
                                        # delc=self.cellsize, top=self.dem, botm=botm, perlen=nstp, nstp=nstp, steady=steady, start_datetime='')
#Flags
Allextentflag=False
Offlineextentflag=False
PRMSOnly=True
# Coordinates of the model extent
xll = 466500      # Lower Left x coordiante    = x Upper Left
yll = 4312800     # Lower Left y coordiante
xur = 500800      # Upper right x coordiante
yur = 4361100     # New Upper right y coordiante    =  y Upper Left
cellsize = 100

# Grid and Layer definition
nlay = 3
nrow = int((yur - yll)/ cellsize)
ncol = int((xur-xll)/cellsize)

# Load Layers
if Offlineextentflag:
    dem = np.loadtxt('Input_Data\\ASCI_Files_June\\DEM.asc', skiprows =0)        # Digital elevation model of the basin.
    th1 = np.loadtxt('Input_Data\\ASCI_Files_June\\th1all.asc', skiprows = 0)
    th1=th1+5
    th2 = np.loadtxt('Input_Data\\ASCI_Files_June\\th2all.asc', skiprows = 0)
    th2=th2+0.1               # thikness of older alluvium.
    th3 = np.loadtxt('Input_Data\\ASCI_Files_June\\th3all.asc', skiprows = 0)
    th3=th3+0.1                # thikness of continental deposit.
    #th4 = 50

elif Allextentflag:
    dem = np.loadtxt('Input_Data\\ASCI_Files_June\\DEM.asc', skiprows =0)        # Digital elevation model of the basin.
    th1 = np.loadtxt('Input_Data\\ASCI_Files_June\\th1all.asc', skiprows = 0)
    th1=th1+5
    th2 = np.loadtxt('Input_Data\\ASCI_Files_June\\th2all.asc', skiprows = 0)
    th2=th2+0.1               # thikness of older alluvium.
    th3 = np.loadtxt('Input_Data\\ASCI_Files_June\\th3all.asc', skiprows = 0)
    th3=th3+0.1                # thikness of continental deposit.
    #th4 = 50

elif PRMSOnly:
    dem = np.loadtxt('Input_Data\\ASCI_Files_June\\DEM.asc', skiprows =0)        # Digital elevation model of the basin.
    th1 = np.loadtxt('Input_Data\\ASCI_Files_June\\th1all.asc', skiprows = 0)
    th1=th1+5
    th2 = np.loadtxt('Input_Data\\ASCI_Files_June\\th2all.asc', skiprows = 0)
    th2=th2+0.1               # thikness of older alluvium.
    th3 = np.loadtxt('Input_Data\\ASCI_Files_June\\th3all.asc', skiprows = 0)
    th3=th3+0.1                # thikness of continental deposit.
    #th4 = 50

else:
    dem = np.loadtxt('Input_Data\\ASCI_Files_June\\DEM.asc', skiprows =0)        # Digital elevation model of the basin.
    th1 = np.loadtxt('Input_Data\\ASCI_Files_June\\th1act.asc', skiprows = 0)
    th1=th1+0.01
    th2 = np.loadtxt('Input_Data\\ASCI_Files_June\\th2act.asc', skiprows = 0)
    th2=th2+0.01               # thikness of older alluvium.
    th3 = np.loadtxt('Input_Data\\ASCI_Files_June\\th3act.asc', skiprows = 0)
    th3=th3+0.01                # thikness of continental deposit.
    th4 = 50

# Calculate the elevations of the models 3 layer
L1botm = dem - th1                         #Bottom Elevation of layer 1
L2botm = L1botm - th2                      #Bottom Elevation of layer 2
L3botm =  L2botm - th3                     #Bottom Elevation of layer 3
#L4botm =  L3botm - th4                     #Bottom Elevation of layer 3
botm = [L1botm, L2botm, L3botm]

# Time discretization
## 1. Transient Model
##### Change Starting Year for different scenarios
STRT_YEAR = 2014
END_YEAR = 2019
per = (END_YEAR - STRT_YEAR)*12 # Number of stress periods
# Assign the number of time steps in each stress period
nstp = []                       # Number of time steps
for y in range(STRT_YEAR,END_YEAR):
    for m in range(1,13):
#       nstp.append(calendar.monthrange(y,m)[1])
       nstp.append(3)

nstp = np.array(nstp)        # Convert from list to array
steady = np.zeros((per),dtype=bool)  # Assign stress period as transient
## 2. Steady Model
#nper = 1
#nstp = 1
#steady = np.ones((nper),dtype=bool)

# Add dis package to Modflow
##### change start_datetime
dis = flopy.modflow.ModflowDis(mf, nlay= nlay, nrow= nrow, ncol= ncol,nper=per, delr=cellsize, delc= cellsize,
                               top=dem, botm=botm,perlen=nstp, nstp=nstp, steady=steady, start_datetime='01\\01\\1991')



#%% 2. Model Boundaries & initial conditions

# Active areas of the 3 layers
# Load the active data with the Inactive Lake area

if Offlineextentflag:
    active1 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer4_PVPOffline.asc', skiprows =0)
    active2 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer2_PVPOffline.asc', skiprows =0)
    active3 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer3_PVPOffline.asc',skiprows =0)
    #active4 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer4_PVPOffline.asc', skiprows =0)

elif Allextentflag:
    active1 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer1_All.asc', skiprows =0)
    active2 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer2_All.asc', skiprows =0)
    active3 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer3_All.asc',skiprows =0)
    #active4 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer4_All.asc', skiprows =0)

elif PRMSOnly:
    active1 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer4_PVPOffline.asc', skiprows =0)
    active2 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer2_OnlyAct.asc', skiprows =0)
    active3 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer3_OnlyAct.asc',skiprows =0)
    vks1act=np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer1_OnlyAct.asc', skiprows =0)
    PRMSOnlyact=np.loadtxt('Input_Data\\ASCI_Files_June\\Active_PRMSOnly.asc', skiprows =0)
else:
    active1 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer1_OnlyAct.asc', skiprows =0)
    active2 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer2_OnlyAct.asc', skiprows =0)
    active3 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer3_OnlyAct.asc',skiprows =0)
    active4 = np.loadtxt('Input_Data\\ASCI_Files_June\\Active_Layer4_All.asc', skiprows =0)

#initialh1=np.loadtxt('Input_Data\\ASCI_Files_June\\Initialheads_1.asc', skiprows =0)
#initialh2=np.loadtxt('Input_Data\\ASCI_Files_June\\Initialheads_2.asc', skiprows =0)
#initialh3=np.loadtxt('Input_Data\\ASCI_Files_June\\Initialheads_3.asc', skiprows =0)
#initialh4=np.loadtxt('Input_Data\\ASCI_Files_June\\Initialheads_4.asc', skiprows =0)

initialh1=np.loadtxt('Input_Data\\ASCI_Files_June\\Initialheads_1_new.asc', skiprows =0)
initialh2=np.loadtxt('Input_Data\\ASCI_Files_June\\Initialheads_2_new.asc', skiprows =0)
initialh3=np.loadtxt('Input_Data\\ASCI_Files_June\\Initialheads_3_new.asc', skiprows =0)

# Load the active data with the Active Lake area
#active = np.loadtxt('Input_Data\\ASCI_Files\\Active_WLake.asc', skiprows =6)


# Define the matrix of the active areas
ibound = [active1, active2,active3]

# Initial head
#strt= np.array([dem]*4)
strtheads= [initialh1, initialh2,initialh3]
#strtheads= [(dem-th1/2), (L1botm-th2/2), (L2botm-th3/2)]

# Initial head from 1/1/2014
#strt = np.load('Input_Data\\initial_conditions\\new_hds_01012014.npy')

# Add BAS package to Modflow
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strtheads)



#%% 3. Aquifer parameterisation
# Load geologic formations
if Offlineextentflag:
    GEO_LYR1 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer1_All.asc',skiprows=0) # Geologic formation distribution in layer 1
    GEO_LYR2 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer2_All.asc',skiprows=0) # Geologic formation distribution in layer 2
    GEO_LYR3 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer3_All.asc',skiprows=0) # Geologic formation distribution in layer 3
    #GEO_LYR4 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer4_All.asc',skiprows=0) # Geologic formation distribution in layer 4

elif Allextentflag:
    GEO_LYR1 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer1_All.asc',skiprows=0) # Geologic formation distribution in layer 1
    GEO_LYR2 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer2_All.asc',skiprows=0) # Geologic formation distribution in layer 2
    GEO_LYR3 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer3_All.asc',skiprows=0) # Geologic formation distribution in layer 3
    #GEO_LYR4 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer4_All.asc',skiprows=0) # Geologic formation distribution in layer 4

elif PRMSOnly:
    GEO_LYR1 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer1_All.asc',skiprows=0) # Geologic formation distribution in layer 1
    GEO_LYR2 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer2_All.asc',skiprows=0) # Geologic formation distribution in layer 2
    GEO_LYR3 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer3_All.asc',skiprows=0) # Geologic formation distribution in layer 3

else:
    GEO_LYR1 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer1_OnlyAct.asc',skiprows=0) # Geologic formation distribution in layer 1
    GEO_LYR2 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer2_OnlyAct.asc',skiprows=0) # Geologic formation distribution in layer 2
    GEO_LYR3 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer3_OnlyAct.asc',skiprows=0) # Geologic formation distribution in layer 3
    GEO_LYR4 = np.loadtxt('Input_Data\\ASCI_Files_June\\Zones_Layer4_OnlyAct.asc',skiprows=0) # Geologic formation distribution in layer 4

GEO = [GEO_LYR1, GEO_LYR2,GEO_LYR3]




#### Change aquifer parameters

hk1_1  =  10.00000                 # Layer 1 tributaries
hk1_2  =  30.00000                # Layer 1 Layer area from lake to drainage
hk1_3  =  30.00000               # Layer 1 Layer area Redwood valley up to lake
vk1_1  =  10.00000
vk1_2  =  30.00000
vk1_3  =  30.00000
ss1_1  =  .0010000
ss1_2  =  .0020000
ss1_3  =  .0020000
if (Allextentflag or Offlineextentflag or PRMSOnly):
    hk1_4  =  0.0100                    # Layer 1 PRMS area (do not calibrate)
    vk1_4  =  hk1_4*0.1
    ss1_4  =  0.0001

hk2_1  =  .30000000               # Layer 2 below layer 1
hk2_2  =  .30000000              # Layer 2 Layer area outside layer 1
vk2_1  =  .30000000
vk2_2  =  .30000000
ss2_1  =  .00100000
ss2_2  =  .00100000
if (Allextentflag or Offlineextentflag or PRMSOnly):
    hk2_3  =  hk1_4                    # Layer 2 PRMS area (do not calibrate)
    vk2_3  =  hk2_3
    ss2_3  =   0.0001

hk3_1  =  .05000000              # Layer 3 below layer 2
hk3_2  =  .05000000             # Layer 3 layer extent within basin outsilde layer 2
vk3_1  =  .05000000
vk3_2  =  .05000000
ss3_1  =  .00400000
ss3_2  =  .00600000
if (Allextentflag or Offlineextentflag or PRMSOnly):
    hk3_3  =  1.20000       # Layer 3 PRMS area (do not calibrate)
    vk3_3  =  hk3_3
    ss3_3  =  0.0001

    hk3_4  =  hk1_1                   # Layer 3 inactive useless
    vk3_4  =  hk3_4
    ss3_4  =   0.0001

#hk4  = 1.0E-5          # Bedrock
#vk4  = hk4
#ss4  =  0.00001

if Offlineextentflag or PRMSOnly:
    ## Organize geologic parameters
    frmt1 = [0,1,2,3,4] # Number of geologic formations in layer 1
    frmt2 = [5,6,7,8] # Number of geologic formations in layer 2
    frmt3 = [9,10,11,12] # Number of geologic formations in layer 3
    #frmt4 = [13,14] # Number of geologic formations in layer 4

    ZoneParams = {}
    # Zone hydraulic conductivity
    ZoneParams['HK'] = [[frmt1, frmt2,frmt3],[0,hk1_1,hk1_2,hk1_3,hk1_4, 0,hk2_1,hk2_2,hk2_3, 0,hk3_1,hk3_2,hk3_3,hk3_4 ]] # m/d

    # Zone hydraulic conductivity vertical anisotropy
    ZoneParams['VKA'] = [[frmt1, frmt2,frmt3],[0,vk1_1,vk1_2,vk1_3,vk1_4, 0,vk2_1,vk2_2,vk2_3, 0,vk3_1,vk3_2,vk3_3,vk3_4 ]]

    # Zone specific storage
    ZoneParams['SS'] = [[frmt1, frmt2,frmt3],[0,ss1_1,ss1_2,ss1_3,ss1_4, 0,ss2_1,ss2_2,ss2_3, 0,ss3_1,ss3_2,ss3_3,ss3_4]]  # 1/m

elif Allextentflag:
    ## Organize geologic parameters
    frmt1 = [0,1,2,3,4] # Number of geologic formations in layer 1
    frmt2 = [5,6,7,8] # Number of geologic formations in layer 2
    frmt3 = [9,10,11,12] # Number of geologic formations in layer 3
    #frmt4 = [13,14] # Number of geologic formations in layer 4

    ZoneParams = {}
    # Zone hydraulic conductivity
    ZoneParams['HK'] = [[frmt1, frmt2,frmt3],[0,hk1_1,hk1_2,hk1_3,hk1_4, 0,hk2_1,hk2_2,hk2_3, 0,hk3_1,hk3_2,hk3_3,hk3_4 ]] # m/d

    # Zone hydraulic conductivity vertical anisotropy
    ZoneParams['VKA'] = [[frmt1, frmt2,frmt3],[0,vk1_1,vk1_2,vk1_3,vk1_4, 0,vk2_1,vk2_2,vk2_3, 0,vk3_1,vk3_2,vk3_3,vk3_4 ]]

    # Zone specific storage
    ZoneParams['SS'] = [[frmt1, frmt2,frmt3],[0,ss1_1,ss1_2,ss1_3,ss1_4, 0,ss2_1,ss2_2,ss2_3, 0,ss3_1,ss3_2,ss3_3,ss3_4]]  # 1/m
else:
    ## Organize geologic parameters
    frmt1 = [0,1,2,3] # Number of geologic formations in layer 1
    frmt2 = [4,5,6] # Number of geologic formations in layer 2
    frmt3 = [7,8,9] # Number of geologic formations in layer 3
    frmt4 = [10,11] # Number of geologic formations in layer 4

    ZoneParams = {}
    # Zone hydraulic conductivity
    ZoneParams['HK'] = [[frmt1, frmt2,frmt3,frmt4],[0,hk1_1,hk1_2,hk1_3, 0,hk2_1,hk2_2, 0,hk3_1,hk3_2,hk3_3, 0,hk4 ]] # m/d

    # Zone hydraulic conductivity vertical anisotropy
    ZoneParams['VKA'] = [[frmt1, frmt2,frmt3,frmt4],[0,vk1_1,vk1_2,vk1_3, 0,vk2_1,vk2_2, 0,vk3_1,vk3_2,vk3_3, 0,vk4 ]]

    # Zone specific storage
    ZoneParams['SS'] = [[frmt1, frmt2,frmt3,frmt4],[0,ss1_1,ss1_2,ss1_3, 0,ss2_1,ss2_2, 0,ss3_1,ss3_2,ss3_3, 0,ss4 ]]  # 1/m

# Layer properties Zones
HK = np.zeros((nlay,nrow,ncol)) # Hydraulic conductivity
VKA = np.zeros((nlay,nrow,ncol)) # Vertical anisotropy (H:V) of hydraulic conductivity
SS = np.zeros((nlay,nrow,ncol)) # Specific storage


# Loop through the layers and formations for each layer to apply the geologic parameters to each array
for l in range(nlay):

    for f in ZoneParams['HK'][0][l]:
        HK[l,:,:] += (GEO[l] == f) * ZoneParams['HK'][1][f]

        # Vertical anisotropy (H:V) of hydraulic conductivity
    for f in ZoneParams['VKA'][0][l]:
        VKA[l,:,:] += (GEO[l] == f) * ZoneParams['VKA'][1][f]

        # Specific storage
    for f in ZoneParams['SS'][0][l]:
        if l==0:
            SS[l,:,:] += (GEO[l] == f) * ZoneParams['SS'][1][f]/th1
        elif l==1:
            SS[l,:,:] += (GEO[l] == f) * ZoneParams['SS'][1][f]/th2
        elif l==2:
            SS[l,:,:] += (GEO[l] == f) * ZoneParams['SS'][1][f]/th3
        else:
            SS[l,:,:] += (GEO[l] == f) * ZoneParams['SS'][1][f]/th4


#layvka = [1]*nlay # Indicates that VKA represents the ratio of H:V hydraulic conductivity
# Layer type 1: Convertible 0: Confined
laytyp=np.array([0,0,0])

lpf = flopy.modflow.ModflowLpf(mf, ipakcb=53, laytyp=laytyp, hk=HK, vka=VKA, ss=SS)


#%%  4. Rvier Boundary condition
# SFR Package
# Load reach and segment data_Normal
#reach_data= pd.read_csv('Input_Data\\SFR_New\\With_Diversion\\Data2_NewSFR_Diversion_AMFeb.csv')

# Load reach and segment data_Strhk Modified
reach_data= pd.read_csv('Input_Data\\SFR_New\\With_Diversion\\Data2_NewSFR_Diversion_AMFeb.csv')

##### Change Str Params
strhc1org=0.001
strhc2org=0.003
strhc3org=0.01
strhc4org=0.03
strhc5org=0.0  #Do not calibrate for diversions
strhc1=0.01
strhc2=0.03
strhc3=0.1
strhc4=0.3
strhc5=0.0  #Do not calibrate for diversions

reach_data.strhc1 = reach_data.strhc1.replace({strhc1org: strhc1,
                                               strhc2org: strhc2,
                                               strhc3org: strhc3,
                                               strhc4org: strhc4,
                                               strhc5org: strhc5})


segment_data = pd.read_csv('Input_Data\\SFR_New\\With_Diversion\\SI_1991-2018_Baseline.csv')


# Convert df to recarray "records of array"
reach_data = reach_data.to_records(index= False)
segment_data = segment_data.to_records(index= False)

# Number of stream reaches (finite-difference cells)
nstrm =-4596
# Nmber of stream segments
nss= 230


#
segment_data_dict = {}
for i in range(0,per):
    segment_data_dict[i]= segment_data[(i*nss ):(i*nss)+nss]


# Data Set 5
dataset_5_dic = {}
for i in range (0,per):
    dataset_5_dic[i]=[nss,0,0]

sfr = flopy.modflow.mfsfr2.ModflowSfr2(mf, nstrm=nstrm, nss= nss, const=86400, dleak=0.0001,isfropt= 1, irtflg=0, reach_data=reach_data,
                                      dataset_5 =dataset_5_dic, ipakcb=53,istcb2=555, segment_data=segment_data_dict,
                                      extension='sfr')

#%% 6. General head boundary condition
##Build Dictionnary to define GHB for each stress period {Nstress Period:[lay, row, col, head, Conductance]}
#ghb_dic = {}
#
#
#
# Dictionnary of GHB
#function to make the list for any stress period
def GHB(Head ):
    row= 409
    col_S= 215
    col_E= 222
    col = np.arange(col_S, col_E, 1)
    ncol = len(col)
    G1 = 0.05          #Conductance
    Head_sp = Head [row, col]
    Ghb_sp = []
    nla =3
    for i in range(0,ncol):
        for j in range (0,nla):
            Ghb_sp.append([j, row, col[i], Head_sp[i], G1])
    return Ghb_sp
# List of all the stress periods
Ghb_tot = GHB (dem)


#per = 3
# Populating the dictionnary
GHB_Dict = {}
keys = range(0,per)
for i in  range(0,per):
        GHB_Dict[i] = Ghb_tot


ghb = flopy.modflow.ModflowGhb(mf, ipakcb=53, stress_period_data=GHB_Dict)


#%% 7.5 Ag wells SRM 121019
# Well package

# Function to open the ag. well data from a single spreadsheet


### 1991 Hist
Wel_Data =  pd.read_csv('Input_Data\\Wells\\SI20211217_1991-01-01_GW_Pump_Mun_Ag_cubicm.csv')* 1.0000000
Wel_Data = Wel_Data* -1
Wel_Loc = pd.read_csv('Input_Data\\Wells\\AM20210603_1991-01-01_Well_Loc_Mun_Ag.csv')


Welrow = list(Wel_Loc['Row'])
Welcol = list(Wel_Loc['Col'])
Wells_Pump = list(Wel_Data.columns)
W = len(Wel_Loc)  # Number of wells

nla =3         # Nb of layers within the wells
New_WEL = []
for perw in  range(0,per):
    for w in range(0,W):
        pump_rate_now = Wel_Data.iloc[[perw],[w]]
        New_WEL.append([1, Welrow[w], Welcol[w], pump_rate_now.values])

# Populating the dictionnary  Structure {Nstress Period:[[lay, row, col, flux], [lay, row, col, flux], [lay, row, col, flux] ],}
WEL_Dict = {}
values = New_WEL
for i in range(0,per):
    WEL_Dict[i] = values[(i*W ):(i*W)+W]


wel = flopy.modflow.mfwel.ModflowWel(mf, ipakcb=53, stress_period_data=WEL_Dict)


#%% 8. Recharge boundary condition
# SRM 120919

#read in recharge array text files
rech_dict = {}

# convert starting year to array numbering
#per_tot = 335
#strt_tot = 276

# convert starting year to array numbering
strt_rech = 1
end_rech = 336

for i in range(strt_rech,end_rech):
    rech_dict[i] = np.loadtxt('Input_Data\\Rechare_Baseline\\Rech_'+str(i+312)+'.asc')*0.2 * 1.000000000

iuzfbnd=active1

rch= flopy.modflow.ModflowRch(mf, nrchop=2, ipakcb=53, rech=rech_dict, irch=iuzfbnd)
#rch= flopy.modflow.mfrch.ModflowRch(mf, nrchop=3, ipakcb=53, rech=rech_dict, irch=1)
                  uzgag=None, extension='uzf', unitnumber=None, filenames=None, options=None, surfk=surfkvks)
#%% 9. heads observations
# HOB package

#### Change based on the Starting year
hobs = flopy.modflow.ModflowHob.load('Input_Data\\Observations\\Ukiah_June_fast_1991.hob', mf)



#%% 10. gage observations
#
#
#gages = [[206,56,400,2],] # Segment,Reach, unit, output type
gages = [[208,58,400,2],[157,30,400,2],[43,9,400,2]]  # Hopland / Talmage
gages = np.array(gages)
#filenames = ['Hopland_div.gag', 'Talmage_div.gag','WestFork_div.gag']
#files = ['D:\\A_Ukiah_SA\\Local_SA\\Flopy_Outputs_SA', 'D:\\A_Ukiah_SA\\Local_SA\\Flopy_Outputs_SA',
#             'D:\\A_Ukiah_SA\\Local_SA\\Flopy_Outputs_SA']

filenames = ['gagesI', 'gageO']
files = ['D:\\A_Ukiah_SA\\Local_SA\\','D:\\A_Ukiah_SA\\Local_SA\\Flopy_Outputs_SA','D:\\A_Ukiah_SA\\Local_SA\\Flopy_Outputs_SA' ]

gage = flopy.modflow.ModflowGage(mf, numgage=3, gage_data=gages, files=files, filenames=filenames)

##gage = flopy.modflow.ModflowGage.load('Flopy_Outputs\\outlet.gag', mf)
#

#%%# Add OC and PCG package to the MODFLOW model :  Output Contorl (what and when information is to be output)
#spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']} # tuple of stress period
#oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

spd = {}
#data2record1 = ['save head', 'save budget', 'print budget'] ##SRM: orignal setup by Samira. creates massive output (>100Gb, 4hr21min runtime)
data2record1 = ['save head', 'save budget', 'print budget' ]
data2record2 = ['save budget']
data2record3= [ 'print budget' ]
#data2record = ['save head'] ##SRM: new, more compact setup
for y in range(STRT_YEAR,END_YEAR):
    for m in range(1,13):
        for d in range(0,calendar.monthrange(y,m)[1]):
            #if d==((calendar.monthrange(y,m)[1])-1) and y==(END_YEAR-1) and m==12:
                #spd[(y-STRT_YEAR)*12 + m - 1, d] = data2record3.copy()

            if d==0:
                spd[(y-STRT_YEAR)*12 + m - 1, d] = data2record1.copy()
            else:
                spd[(y-STRT_YEAR)*12 + m - 1, d] =data2record2.copy()


            #spd[((y-1)*12) + m, d+1] = data2record.copy()
#    spd[14,30] = ['save head', 'save drawdown', 'save budget', 'print budget', 'ddreference']
#    for p in [6,20,32,41,54,78,90,102,114,128,138,150,162,175,187,198,213,225,235]:
#        spd[p,0] = data2record.copy()

oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

# Specify the solver PCG
#pcg = flopy.modflow.ModflowPcg(mf)dem = np.loadtxt('Input_Data\\ASCI_Files\\dem.asc', skiprows =6)
pcg = flopy.modflow.ModflowPcg(mf, mxiter=2000,iter1=100,hclose=0.001, rclose=0.1)

#
#nwt = flopy.modflow.mfnwt.ModflowNwt(mf,headtol=0.001, fluxtol=5000, maxiterout=2000, thickfact=1e-5, linmeth=1, iprnwt=2,
#                              ibotav=0, options='SPECIFIED', Continue=True, dbdtheta=0.3, dbdkappa=1e-6, dbdgamma=0.0001,
#                            momfact=0.001, backflag=0, maxbackiter=10, backtol=1, backreduce=0.5, maxitinner=200,
#                             ilumethod=2, levfill=1, stoptol=1e-15, msdr=10, iacl=2, norder=1, level=5, north=7, iredsys=0,
#                            rrctols=0.0, idroptol=1, epsrn=0.0001, hclosexmd=0.0001, mxiterxmd=50, extension='nwt',
#                           unitnumber=None, filenames=None)

# Write the MODFLOW model input files
mf.write_input()

#Run the MODFLOW model
success, buff = mf.run_model()





