#Install
!pip install lightkurve
!pip install sktime
#
#
#Google Drive
from google.colab import drive
drive.mount('/content/drive')
#
#
#Data Acquisition
book_local = 'shallue_local_curves_'
book_global = 'shallue_global_curves_'
path_input = "/content/drive/MyDrive/Iniciação Científica/IC_Exoplanetas_2022_Experimento/Base de Dados/lighkurve_KOI_dataset.csv"
path_local = '/content/drive/MyDrive/Iniciação Científica/IC_Exoplanetas_2022_Experimento/Base de Dados/Resultados teste/' + book_local + '.xlsx'
path_global = '/content/drive/MyDrive/Iniciação Científica/IC_Exoplanetas_2022_Experimento/Base de Dados/Resultados teste/' + book_global + '.xlsx'

import pandas as pd
import numpy as np
import time
from lightkurve import search_lightcurve 

lc = pd.read_csv(path_input, sep = ",") 
lc = lc[['kepid','koi_disposition','koi_period','koi_time0bk','koi_duration','koi_quarters']]

lc.shape
print('total inicial de curvas: %d\n'%(lc.shape[0]))

lc = lc.dropna()
lc = lc[lc.koi_disposition != 'CANDIDATE']
lc = lc.reset_index(drop=True)
print('falsos positivos: %d, confirmados: %d\n\ntotal atualizado: %d\n'%((lc.koi_disposition == 'FALSE POSITIVE').sum(),(lc.koi_disposition == 'CONFIRMED').sum(),lc.shape[0]))

perc_class = ((lc.koi_disposition == 'FALSE POSITIVE').sum()*100)/lc.shape[0]
print('falsos positivos: %.2f %% confirmados: %.2f %% \n'%(perc_class,100-perc_class))

import sys
import warnings
warnings.simplefilter("ignore")

curvas_locais = []
labels_locais = []
curvas_globais = []
labels_globais = []
start_time = time.time()
for index, row in lc[5000:6000].iterrows():
  period, t0, duration_hours = row[2], row[3], row[4]

  try:

    lcs = search_lightcurve(str(row[0]), author='Kepler', cadence='long').download_all()

    if (lcs != None):

      lc_raw = lcs.stitch()
      lc_raw.flux.shape

      lc_clean = lc_raw.remove_outliers(sigma=3)

      temp_fold = lc_clean.fold(period, epoch_time=t0)
      fractional_duration = (duration_hours / 24.0) / period
      phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
      transit_mask = np.in1d(lc_clean.time.value, temp_fold.time_original.value[phase_mask])

      lc_flat, trend_lc = lc_clean.flatten(return_trend=True, mask=transit_mask)

      lc_fold = lc_flat.fold(period, epoch_time=t0)

      #global preprocessing-----------------------------------------------------
      lc_global = lc_fold.bin(bins=2001).normalize() - 1
      lc_global = (lc_global / np.abs(np.nanmin(lc_global.flux)) ) * 2.0 + 1
      lc_global.flux.shape
      #global preprocessing-----------------------------------------------------
    
      phase_mask = (lc_fold.phase > -4*fractional_duration) & (lc_fold.phase < 4.0*fractional_duration)
      lc_zoom = lc_fold[phase_mask]

      #local preprocessing------------------------------------------------------
      lc_local = lc_zoom.bin(bins=201).normalize() - 1
      lc_local = (lc_local / np.abs(np.nanmin(lc_local.flux)) ) * 2.0 + 1
      lc_local.flux.shape
      #local--------------------------------------------------------------------

      labels_locais.append(row[1])
      curvas_locais.append(lc_local.flux.value)

      labels_globais.append(row[1])
      curvas_globais.append(lc_global.flux.value)

      print(index, 'OK')

    else:
      print(index, 'not downloaded')  
    
  except Exception as e:
    print(index, e)

t = time.time() - start_time   

print('Tempo para importar curvas de luz: %f seconds\n' %t)

dataset_global = pd.DataFrame(curvas_globais)
dataset_local = pd.DataFrame(curvas_locais)

for i in range(1,len(curvas_globais)):
    if len(curvas_globais[i]) != len(curvas_globais[i-1]):
        print("A curva %d possui tamanho diferente das demais curvas. Tamanho: %d"%(i,len(curvas_globais[i])))
print("Caso nenhuma das curvas apresente tamanho diferente, todas as curvas GLOBAIS possuem o total de %d pontos cada.\n"%len(curvas_globais[0]))

for i in range(1,len(curvas_locais)):
    if len(curvas_locais[i]) != len(curvas_locais[i-1]):
        print("A curva %d possui tamanho diferente das demais curvas. Tamanho: %d"%(i,len(curvas_locais[i])))
print("Caso nenhuma das curvas apresente tamanho diferente, todas as curvas LOCAIS possuem o total de %d pontos cada.\n"%len(curvas_locais[0]))

print("Quantidade de NaN na base GLOBAL: %s"%dataset_global.isna().sum(axis=1).sum())
print("Quantidade de NaN na base LOCAL: %s\n"%dataset_local.isna().sum(axis=1).sum())

perc_nan_glob = (dataset_global.isna().sum(axis=1).sum()*100)/dataset_global.count(axis=1).sum()
print("Porcentagem de valores do dataset GLOBAL substituídos na interpolação: %.2f %%"%perc_nan_glob)
perc_nan_loc = (dataset_local.isna().sum(axis=1).sum()*100)/dataset_local.count(axis=1).sum()
print("Porcentagem de valores do dataset LOCAL substituídos na interpolação: %.2f %%"%perc_nan_loc)

dataset_global = dataset_global.interpolate(axis=1)
dataset_local = dataset_local.interpolate(axis=1)

print("Quantidade de NaN na base GLOBAL após interpolação: %s --> deve sempre ser zero"%dataset_global.isna().sum(axis=1).sum())
print("Quantidade de NaN na base LOCAL após interpolação: %s --> deve sempre ser zero"%dataset_local.isna().sum(axis=1).sum())

labels_glob = pd.Series(labels_globais)
labels_loc = pd.Series(labels_locais)
dataset_global['label'] = labels_glob
dataset_local['label'] = labels_loc

dataset_global.to_csv(path_global,index=False)  
dataset_local.to_csv(path_local,index=False)
