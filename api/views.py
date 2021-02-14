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


def mag(df,x,y,z):
    m = (df[x]**2+df[y]**2+df[z]**2)**0.5
    return m
def agv(df,gx,gy,gz,lx,ly,lz,gmod):
    return (df[gx]*df[lx]+df[gy]*df[ly]+df[gz]*df[lz])/(df[gmod]*df[gmod])

def predict_both(fol,path):
	model = joblib.load(path+"fall_predict_both.pkl")
	#initializing values
	fol['l_x'] = fol['acc_x'] - fol['grav_x']
	fol['l_y'] = fol['acc_y'] - fol['grav_y']
	fol['l_z'] = fol['acc_z'] - fol['grav_z']
	fol['acc_mag'] = mag(fol,"acc_x","acc_y","acc_z")
	fol['grav_mag'] = mag(fol,"grav_x","grav_y","grav_z")
	fol['l_mag'] = mag(fol,"l_x","l_y","l_z")
	fol['gyro_mag'] = mag(fol,"gyro_x","gyro_y","gyro_z")

	fol['agv'] = agv(fol,'grav_x','grav_y','grav_z','l_x','l_y','l_z','grav_mag')

	df = pd.DataFrame()
	
	t_s = fol.iloc[0]['rel_time']
	t_e = fol.iloc[fol.shape[0]-1]['rel_time']
	t_s = int(t_s)
	t_e = int(t_e)
	for t in range(t_s,t_e+1):
	    y = fol[(fol['rel_time']>=t) & (fol['rel_time']<(t+1))]
	    
	    size = y.shape[0]
	    start = y.index[0]
	    end = y.index[0]+size
	    time = t
	    acc_max = y['acc_mag'].max()
	    acc_min = y['acc_mag'].min()
	    acc_mean = y['acc_mag'].mean()
	    agv_max = y['agv'].max()
	    agv_min = y['agv'].min()
	    lin_max = y['l_mag'].max()
	    lin_min = y['l_mag'].min()
	    lin_mean = y['l_mag'].mean()
	    grav_max = y['grav_mag'].max()
	    grav_min = y['grav_mag'].min()
	    gyro_max = y['gyro_mag'].max()
	    gyro_min = y['gyro_mag'].min()

	    df = df.append({'time':time,'acc_max':acc_max,'acc_min':acc_min,'acc_mean':acc_mean,'agv_max':agv_max,'agv_min':agv_min,'lin_max':lin_max,'lin_min':lin_min,'grav_max':grav_max,'grav_min':grav_min,'gyro_max':gyro_max,'gyro_min':gyro_min},ignore_index=True)

	i = df['acc_max'].idxmax()
	acc_max = df.iloc[i]['acc_max']

	if(i>0):
	    if((df.iloc[i-1]['agv_max']>df.iloc[i]['agv_max']) & (df.iloc[i-1]['agv_min']<df.iloc[i]['agv_min'])):
	        agv_max = df.iloc[i-1]['agv_max']
	        agv_min = df.iloc[i-1]['agv_min']
	    else:
	        agv_max = df.iloc[i]['agv_max']
	        agv_min = df.iloc[i]['agv_min']
	else:
	    agv_max = df.iloc[i]['agv_max']
	    agv_min = df.iloc[i]['agv_min']

	if(i<=3):
	    f6 = 1 #indicating f7 is present
	    lin_max = df.iloc[i]['lin_max']
	    gyro_max = df.iloc[i]['gyro_max']
	    f7 = df.iloc[i]['lin_max']-df.iloc[i+2]['lin_max']
	    f8 = df.iloc[i]['gyro_max']-df.iloc[i+2]['gyro_max']
	else:
	    f6 = 0 # indicating f7 is None
	    lin_max = df.iloc[i]['lin_max']
	    gyro_max = df.iloc[i]['gyro_max']
	    f7 = None
	    f8 = None

	skew_acc = fol['acc_mag'].skew()
	kurt_acc = fol['acc_mag'].kurtosis()
	skew_gyro = fol['gyro_mag'].skew()
	kurt_gyro = fol['gyro_mag'].kurtosis()

	data={'acc_max':acc_max,'agv_max':agv_max,'agv_min':agv_min,'f7':f7,'f8':f8,'gyro_max':gyro_max,'kurt_acc':kurt_acc,'kurt_gyro':kurt_gyro,'lin_max':lin_max,'skew_acc':skew_acc,'skew_gyro':skew_gyro}
	data = pd.Series(data)
	data = [data]
	print(data)
	ret=0.0
	if(f6==0):
		ret=0.0
	else:
		preds=model.predict(data)
		# if(preds>=0.7):
		# 	ret=preds
		# else:
		# 	ret=preds
		ret = preds

	return ret

def predict_acc(fol,path):
	model = joblib.load(path+"fall_predict_acc.pkl")	
	#initializing values
	fol['l_x'] = fol['acc_x'] - fol['grav_x']
	fol['l_y'] = fol['acc_y'] - fol['grav_y']
	fol['l_z'] = fol['acc_z'] - fol['grav_z']
	fol['acc_mag'] = mag(fol,"acc_x","acc_y","acc_z")
	fol['grav_mag'] = mag(fol,"grav_x","grav_y","grav_z")
	fol['l_mag'] = mag(fol,"l_x","l_y","l_z")
	fol['agv'] = agv(fol,'grav_x','grav_y','grav_z','l_x','l_y','l_z','grav_mag')

	df = pd.DataFrame()
	
	t_s = fol.iloc[0]['rel_time']
	t_e = fol.iloc[fol.shape[0]-1]['rel_time']
	t_s = int(t_s)
	t_e = int(t_e)
	error = False
	for t in range(t_s,t_e+1):
	    y = fol[(fol['rel_time']>=t) & (fol['rel_time']<(t+1))]
	    
	    size = y.shape[0]
	    if(size==0):
	    	error=True
	    	break
	    start = y.index[0]
	    end = y.index[0]+size
	    
	    time = t
	    acc_max = y['acc_mag'].max()
	    acc_min = y['acc_mag'].min()
	    acc_mean = y['acc_mag'].mean()
	    agv_max = y['agv'].max()
	    agv_min = y['agv'].min()
	    lin_max = y['l_mag'].max()
	    lin_min = y['l_mag'].min()
	    lin_mean = y['l_mag'].mean()
	    grav_max = y['grav_mag'].max()
	    grav_min = y['grav_mag'].min()

	    df = df.append({'time':time,'acc_max':acc_max,'acc_min':acc_min,'acc_mean':acc_mean,'agv_max':agv_max,'agv_min':agv_min,'lin_max':lin_max,'lin_min':lin_min,'grav_max':grav_max,'grav_min':grav_min},ignore_index=True)

	i = df['acc_max'].idxmax()
	acc_max = df.iloc[i]['acc_max']
	if(error):
		print("_______________THERE WAS AN ERROR_____________________________")
		return 0

	if(i>0):
	    if((df.iloc[i-1]['agv_max']>df.iloc[i]['agv_max']) & (df.iloc[i-1]['agv_min']<df.iloc[i]['agv_min'])):
	        agv_max = df.iloc[i-1]['agv_max']
	        agv_min = df.iloc[i-1]['agv_min']
	    else:
	        agv_max = df.iloc[i]['agv_max']
	        agv_min = df.iloc[i]['agv_min']
	else:
	    agv_max = df.iloc[i]['agv_max']
	    agv_min = df.iloc[i]['agv_min']

	if(i<=3):
	    f6 = 1 #indicating f5 is present
	    lin_max = df.iloc[i]['lin_max']
	    f7 = df.iloc[i]['lin_max']-df.iloc[i+2]['lin_max']
	else:
	    f6 = 0 # indicating f5 is None
	    lin_max = df.iloc[i]['lin_max']
	    f7 = None
	skew_acc = fol['acc_mag'].skew()
	kurt_acc = fol['acc_mag'].kurtosis()
	data ={'acc_max':acc_max,'agv_max':agv_max,'agv_min':agv_min,'f7':f7,'kurt_acc':kurt_acc,'lin_max':lin_max,'skew_acc':skew_acc}
	data = pd.Series(data)
	data = [data]
	#print(data)
	ret=0.0
	if(f6==0):
		ret=0.0
	else:
		preds=model.predict(data)
		# if(preds>=0.7):
		# 	ret=preds
		# else:
		# 	ret=preds
		ret = preds[0]
	return ret


# Create your views here.
@api_view(["POST"])
def data_list(request):
    if request.method == 'POST':
        data = JSONParser().parse(request)
        df = pd.DataFrame.from_records(data)
        df['rel_time'] = (df['timestamp']-df.loc[0]['timestamp'])/1000.0
       


        has_fall = 0.0

        path = settings.BASE_DIR
        path = str(path)+"\\api\\"

        #print(df.columns.size)
        if(df.columns.size==11):
        	#for both gyro and acc
        	has_fall = predict_both(df,path)
        elif(df.columns.size==8):
        	#for onlt acc
        	has_fall = predict_acc(df,path)
        
        print('response send: {}'.format(has_fall))
        fall = {
        	'fall':has_fall
        }
        return JsonResponse(fall, safe=False)

    elif request.method == 'GET':
        data = Data.objects.all()
        data_serializer = DataSerializer(data, many=True)
        return JsonResponse(data_serializer.data, safe=False)

