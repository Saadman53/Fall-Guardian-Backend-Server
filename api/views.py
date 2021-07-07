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
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import joblib
from django.conf import settings
import pickle as p
import numpy as np
from os import path
from scipy import signal
import os
import math
from scipy.signal import find_peaks, peak_prominences, peak_widths, lfilter


path_n = path.abspath(__file__) # full path of your script
dir_path = path.dirname(path_n) # full path of the directory of your script
file_path1 = path.join(dir_path,'rf_acc.pkl') # pkl file path
model1 = joblib.load(file_path1)

file_path2 = path.join(dir_path,'rf_both.pkl') # pkl file path
model2 = joblib.load(file_path2)

w_acc = [ 0.29532642,-0.05355093,0.15279751,-0.0696737,-0.27507482,0.3590252,0.62624295]
b_acc = 8.397000000000785
w_all = [ 0.31132593,-0.07801587,0.09794796,-0.32345425,-0.07173864,0.13417778,-0.11686499,0.28469475,0.16275261,0.27922792,0.42917821]
b_all = 8.571000000000689

def predict(X,w,b):
	approx = np.dot(X,w)-b
	approx = np.sign(approx)
	approx = np.where(approx==-1,0,1)
	return approx


def mag(df,x,y,z):
    m = (df[x]**2+df[y]**2+df[z]**2)**0.5
    return m
def agv(df,gx,gy,gz,lx,ly,lz,gmod):
    return (df[gx]*df[lx]+df[gy]*df[ly]+df[gz]*df[lz])/(df[gmod]*df[gmod])

def gravlin(fol,acc,length):
    g = [None]*length
    g = np.array(g)
    l = [None]*length
    l = np.array(l)
    g = lfilter([0.2],[1.0,-0.8],fol[acc])
    l = fol[acc]-g
    return pd.Series(g,index = fol.index),pd.Series(l,index = fol.index)


def predict_both(fol):
	#initializing values
	x = fol.copy()
	length = x.shape[0]
	x['grav_x'], x['l_x'] = gravlin(x,"acc_x",length)
	x['grav_y'], x['l_y'] = gravlin(x,"acc_y",length)
	x['grav_z'], x['l_z'] = gravlin(x,"acc_z",length)
	x = x[['acc_x','acc_y','acc_z','grav_x','grav_y','grav_z','l_x','l_y','l_z','gyro_x','gyro_y','gyro_z']]
	x = x.rolling(10,min_periods=1).mean()
	x['rel_time'] = fol['rel_time']
	x['acc_mag'] = mag(x,"acc_x","acc_y","acc_z")
	x['grav_mag'] = mag(x,"grav_x","grav_y","grav_z")
	x['lin_mag'] = mag(x,"l_x","l_y","l_z")
	x['gyro_mag'] = mag(x,"gyro_x","gyro_y","gyro_z")

	x['agv'] = agv(x,'grav_x','grav_y','grav_z','l_x','l_y','l_z','grav_mag')
	max_acc = x['acc_mag'].max()
	max_time = x[x["acc_mag"]==max_acc].iloc[0]['rel_time']
	final_time = x.tail(1).iloc[0]['rel_time']
	start_time = x.head(1).iloc[0]['rel_time']
	time = final_time - start_time
	ret = 0
	if((max_time>3.0) & (max_time<4.0) & (time<7)):
		df = pd.DataFrame()
		t_s = x.loc[0]['rel_time']
		t_e = x.loc[x.shape[0]-1]['rel_time']
		t_s = int(t_s)
		t_e = int(t_e)
		for t in range(t_s,6):
			y = x[(x['rel_time']>=t) & (x['rel_time']<(t+1))]
			i = y['acc_mag'].idxmax()
			acc_max = y['acc_mag'].max()
			agv_max = y['agv'].max()
			agv_min = y['agv'].min()
			lin_max = y['lin_mag'].max()
			gyro_max = y['gyro_mag'].max()
			q = y.loc[i]['acc_y']
			df = df.append({'acc_max':acc_max,'agv_max':agv_max,'agv_min':agv_min,'lin_max':lin_max,'gyro_max':gyro_max,"q":q},ignore_index=True)
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
		y_min = df.iloc[i+2]['q']
		data={'acc_max':acc_max,'agv_max':agv_max,'agv_min':agv_min,'gyro_max':gyro_max,'kurt_acc':kurt_acc,'kurt_gyro':kurt_gyro,'lin_max':lin_max,'skew_acc':skew_acc,'skew_gyro':skew_gyro,"tGyro_max":tGyro_max,"tLin_max":tLin_max}
		data = pd.Series(data)
		data = [data]
		preds=model2.predict(data)
		ret = preds[0]
		if(ret==1):
			if((abs(y_min))>7):
				print("[Not actually a fall]")
				ret = 0
	return {"fall":int(ret)}

	
	

def predict_acc(fol):
	#initializing values
	x = fol.copy()
	length = x.shape[0]
	x['grav_x'], x['l_x'] = gravlin(x,"acc_x",length)
	x['grav_y'], x['l_y'] = gravlin(x,"acc_y",length)
	x['grav_z'], x['l_z'] = gravlin(x,"acc_z",length)
	x = x[['acc_x','acc_y','acc_z','grav_x','grav_y','grav_z','l_x','l_y','l_z']]
	x = x.rolling(10,min_periods=1).mean()
	x['rel_time'] = fol['rel_time']
	x['l_x'] = x['acc_x'] - x['grav_x']
	x['l_y'] = x['acc_y'] - x['grav_y']
	x['l_z'] = x['acc_z'] - x['grav_z']
	x['acc_mag'] = mag(x,"acc_x","acc_y","acc_z")
	x['grav_mag'] = mag(x,"grav_x","grav_y","grav_z")
	x['lin_mag'] = mag(x,"l_x","l_y","l_z")
	x['agv'] = agv(x,'grav_x','grav_y','grav_z','l_x','l_y','l_z','grav_mag')
	max_acc = x['acc_mag'].max()
	max_time = x[x["acc_mag"]==max_acc].iloc[0]['rel_time']
	final_time = x.tail(1).iloc[0]['rel_time']
	start_time = x.head(1).iloc[0]['rel_time']
	time = final_time - start_time
	ret = 0
	print("start time: {}\nend time: {}\nmax_time: {}\nmax_val: {}".format(start_time,final_time,max_time,max_acc))

	if((max_time>3.0) & (max_time<4.0) & (time<7)):
		df = pd.DataFrame()
		t_s = x.loc[0]['rel_time']
		t_e = x.loc[x.shape[0]-1]['rel_time']
		t_s = int(t_s)
		t_e = int(t_e)
		for t in range(t_s,6):
			y = x[(x['rel_time']>=t) & (x['rel_time']<(t+1))]
			i = y['acc_mag'].idxmax()
			acc_max = y['acc_mag'].max()
			agv_max = y['agv'].max()
			agv_min = y['agv'].min()
			lin_max = y['lin_mag'].max()
			q = y.loc[i]['acc_y']
			df = df.append({'acc_max':acc_max,'agv_max':agv_max,'agv_min':agv_min,'lin_max':lin_max,'q':q},ignore_index=True)
		i = df["acc_max"].idxmax()
		acc_max = df.iloc[i]['acc_max']
		lin_max = df.iloc[i]['lin_max']
		if((df.iloc[i-1]['agv_max']>df.iloc[i]['agv_max']) & (df.iloc[i-1]['agv_min']<df.iloc[i]['agv_min'])):
			agv_max = df.iloc[i-1]['agv_max']
			agv_min = df.iloc[i-1]['agv_min']
		else:
			agv_max = df.iloc[i]['agv_max']
			agv_min = df.iloc[i]['agv_min']
		skew_acc = x['acc_mag'].skew()
		kurt_acc = x['acc_mag'].kurtosis()
		tLin_max = df.loc[i]['lin_max'] - df.loc[i+2]['lin_max']
		y_min = df.iloc[i+2]['q']
		data={'acc_max':acc_max,'agv_max':agv_max,'agv_min':agv_min,'kurt_acc':kurt_acc,'lin_max':lin_max,'skew_acc':skew_acc,"tLin_max":tLin_max}
		data = pd.Series(data)
		data = [data]
		#preds=model1.predict(data)
		preds=model1.predict(data)
		ret = preds[0]
		if(ret==1):
			print("fall")
			if((abs(y_min))>7):
				print("[Not actually a fall]")
				ret = 0
			else:
				print("[Fall occured] {}".format(abs(y_min)))
	return {"fall":int(ret)}


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
        if(df.columns.size==8):
        	#for both gyro and acc
        	has_fall = predict_both(df)
        elif(df.columns.size==5):
        	#for onlt acc
        	has_fall = predict_acc(df)
        print('response send: {}'.format(has_fall["fall"]))
        #has_fall = {"fall":int(0)}
        
        return JsonResponse(has_fall, safe=False)

    elif request.method == 'GET':
        data = Data.objects.all()
        data_serializer = DataSerializer(data, many=True)
        return JsonResponse(data_serializer.data, safe=False)

