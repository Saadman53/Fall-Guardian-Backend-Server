from django.shortcuts import render
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Data
from .serializers import DataSerializer
from rest_framework.decorators import api_view
from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser 
# List all stocks or create new one 
#stocks/
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import joblib
from django.conf import settings
import pickle as p
import numpy as np
from os import path
from scipy import signal
import os
from scipy.signal import find_peaks, peak_prominences, peak_widths


path_n = path.abspath(__file__) # full path of your script
dir_path = path.dirname(path_n) # full path of the directory of your script
file_path1 = path.join(dir_path,'fall_predict_acc.pkl') # pkl file path
#model1 = p.load(open(file_path1,'rb'))
model1 = joblib.load(file_path1)
file_path2 = path.join(dir_path,'new_svm.pkl') # pkl file path
model2 = joblib.load(file_path2)



def mag(df,x,y,z):
    m = (df[x]**2+df[y]**2+df[z]**2)**0.5
    return m
def agv(df,gx,gy,gz,lx,ly,lz,gmod):
    return (df[gx]*df[lx]+df[gy]*df[ly]+df[gz]*df[lz])/(df[gmod]*df[gmod])




def predict_both(fol):
	#initializing values
	x = fol.copy()
	x = x[['acc_x','acc_y','acc_z','grav_x','grav_y','grav_z','gyro_x','gyro_y','gyro_z']]
	x = x.rolling(10,min_periods=1).mean()
	x['rel_time'] = fol['rel_time']
	x['l_x'] = x['acc_x'] - x['grav_x']
	x['l_y'] = x['acc_y'] - x['grav_y']
	x['l_z'] = x['acc_z'] - x['grav_z']
	x['acc_mag'] = mag(x,"acc_x","acc_y","acc_z")
	x['grav_mag'] = mag(x,"grav_x","grav_y","grav_z")
	x['lin_mag'] = mag(x,"l_x","l_y","l_z")
	x['gyro_mag'] = mag(x,"gyro_x","gyro_y","gyro_z")

	x['agv'] = agv(x,'grav_x','grav_y','grav_z','l_x','l_y','l_z','grav_mag')
	max_acc = x['acc_mag'].max()
	max_time = x[x["acc_mag"]==max_acc].iloc[0]['rel_time']
	ret = 0
	if((max_time>1.0) & (max_time<4.0)):
		df = pd.DataFrame()
		t_s = x.loc[0]['rel_time']
		t_e = x.loc[x.shape[0]-1]['rel_time']
		t_s = int(t_s)
		t_e = int(t_e)
		for t in range(t_s,6):
			y = x[(x['rel_time']>=t) & (x['rel_time']<(t+1))]
			size = y.shape[0]
			time = t
			acc_max = y['acc_mag'].max()
			acc_min = y['acc_mag'].min()
			acc_mean = y['acc_mag'].mean()
			agv_max = y['agv'].max()
			agv_min = y['agv'].min()
			lin_max = y['lin_mag'].max()
			lin_min = y['lin_mag'].min()
			lin_mean = y['lin_mag'].mean()
			grav_max = y['grav_mag'].max()
			grav_min = y['grav_mag'].min()
			gyro_max = y['gyro_mag'].max()
			gyro_min = y['gyro_mag'].min()
			z_max = max(y['acc_z'].max(),abs(y['acc_z'].min()))
			z_mean = y['acc_z'].mean()
			peaks, _ = find_peaks(y['acc_mag'])
			BPC = (y.iloc[peaks]['acc_mag']>=16.0).sum()
			df = df.append({'time':time,'acc_max':acc_max,'acc_min':acc_min,'acc_mean':acc_mean,'agv_max':agv_max,'agv_min':agv_min,'lin_max':lin_max,'lin_min':lin_min,'grav_max':grav_max,'grav_min':grav_min,'gyro_max':gyro_max,'gyro_min':gyro_min,'z_max':z_max,'z_mean':z_mean,"BPC":BPC},ignore_index=True)
		i = df["acc_max"].idxmax()
		acc_max = df.iloc[i]['acc_max']
		lin_max = df.iloc[i]['lin_max']
		gyro_max = df.iloc[i]['gyro_max']
		if((df.iloc[i-1]['agv_max']>df.iloc[i]['agv_max']) & (df.iloc[i-1]['agv_min']<df.iloc[i]['agv_min'])):
			agv_max = df.iloc[i-1]['agv_max']
			agv_min = df.iloc[i-1]['agv_min']
		else:
			agv_max = df.iloc[i]['agv_max']
			agv_min = df.iloc[i]['agv_min']
		skew_acc = x['acc_mag'].skew()
		kurt_acc = x['acc_mag'].kurtosis()
		skew_gyro = x['gyro_mag'].skew()
		kurt_gyro = x['gyro_mag'].kurtosis()
		tLin_max = df.loc[i]['lin_max'] - df.loc[i+2]['lin_max']
		tGyro_max = df.loc[i]['gyro_max'] - df.loc[i+2]['gyro_max']
		z_max = df.loc[i+2]['z_max']
		z_mean = df.loc[i+2]['z_mean']
		BPC = df.loc[i-1]['BPC'] + df.loc[i]['BPC'] + df.loc[i+1]['BPC']
		data={'acc_max':acc_max,'agv_max':agv_max,'agv_min':agv_min,'gyro_max':gyro_max,'kurt_acc':kurt_acc,'kurt_gyro':kurt_gyro,'lin_max':lin_max,'skew_acc':skew_acc,'skew_gyro':skew_gyro,"tLin_max":tLin_max,"tGyro_max":tGyro_max}
		data = pd.Series(data)
		data = [data]
		preds=model2.predict(data)
		ret = preds[0]
	return {"fall":int(ret)}

	
	

def predict_acc(fol):
	#initializing values
	fol['lin_x'] = fol['acc_x'] - fol['grav_x']
	fol['lin_y'] = fol['acc_y'] - fol['grav_y']
	fol['lin_z'] = fol['acc_z'] - fol['grav_z']
	fol['acc_mag'] = mag(fol,"acc_x","acc_y","acc_z")
	fol['grav_mag'] = mag(fol,"grav_x","grav_y","grav_z")
	fol['lin_mag'] = mag(fol,"lin_x","lin_y","lin_z")

	fol['agv'] = agv(fol,'grav_x','grav_y','grav_z','lin_x','lin_y','lin_z','grav_mag')

	p = fol.tail(1)
	max_acc = max(p['max_acc'].iloc[0],fol['acc_mag'].max())
	min_acc = 0.0
	max_grav = max(p['max_grav'].iloc[0],fol['grav_mag'].max())
	min_grav = 0.0
	max_lin = max(p['max_lin'].iloc[0],fol['lin_mag'].max())
	min_lin = 0.0
	max_agv = max(p['max_agv'].iloc[0],fol['agv'].max())
	min_agv = min(p['min_agv'].iloc[0],fol['agv'].min())
	max_gyro = p['max_gyro'].iloc[0]
	min_gyro = 0.0

	#scaling data
	fol['acc_mag'] = (fol['acc_mag']-min_acc)/(max_acc-min_acc)
	fol['grav_mag'] = (fol['grav_mag']-min_grav)/(max_grav-min_grav)
	fol['lin_mag'] = (fol['lin_mag']-min_lin)/(max_lin-min_lin)
	fol['agv'] = (fol['agv']-min_agv)/(max_agv-min_agv)

	#denoising data
	df = fol.copy()
	df = df[['acc_mag','lin_mag','grav_mag','agv']]
	df = df.rolling(3).mean()
	df['rel_time'] = fol['rel_time']
	#acceleration segment
	acc_max = df['acc_mag'].max()
	acc_std = df['acc_mag'].std()
	acc_var = df['acc_mag'].var()
	#gravity segment
	grav_std = df['grav_mag'].std()
	grav_var = df['grav_mag'].var()
	#linear acceleration segment
	lin_max = df['lin_mag'].max()
	lin_std = df['lin_mag'].std()
	lin_var = df['lin_mag'].var()
	#agv segment
	agv_min = df['agv'].min()
	agv_avg = df['agv'].mean()
	agv_std = df['agv'].std()
	agv_var = df['agv'].var()
	agv_skew = df['agv'].skew()
	agv_kurt = df['agv'].kurtosis()
	data = {'acc_max':acc_max,"acc_std":acc_std,"acc_var":acc_var,"grav_std":grav_std,"grav_var":grav_var,"lin_max":lin_max,"lin_std":lin_std,"lin_var":lin_var,"agv_min":agv_min,"agv_avg":agv_avg,"agv_std":agv_std,"agv_var":agv_var,"agv_skew":agv_skew,"agv_kurt":agv_kurt}
	preds=model2.predict([pd.Series(data)])
	ret = preds[0]
	print(ret)
	res = {'fall':int(ret),"max_acc":max_acc,"min_acc":min_acc,"max_grav":max_grav,"min_grav":min_grav,"max_lin":max_lin,"min_lin":min_lin,"max_agv":max_agv,"min_agv":min_agv,"max_gyro":max_gyro,"min_gyro":min_gyro}
	return res


# Create your views here.
@api_view(["POST"])
def data_list(request):
    if request.method == 'POST':
        data = JSONParser().parse(request)
        df = pd.DataFrame.from_records(data)
        df['rel_time'] = (df['timestamp']-df.loc[0]['timestamp'])/1000.0
       
        #print("col length{}".format(df.shape[1]))

        has_fall = 0.0
        #print(df.columns.size)
        # if(df.columns.size==11):
        # 	#for both gyro and acc
        # 	has_fall = predict_both(df)
        # elif(df.columns.size==8):
        # 	#for onlt acc
        # 	has_fall = predict_acc(df)
        has_fall = predict_both(df)
        print('response send: {}'.format(has_fall["fall"]))
        
        return JsonResponse(has_fall, safe=False)

    elif request.method == 'GET':
        data = Data.objects.all()
        data_serializer = DataSerializer(data, many=True)
        return JsonResponse(data_serializer.data, safe=False)

