from __future__ import print_function
import os
from os import listdir
import shutil
import csv
import sys
import errno
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

''' NOTE
    -> script is organized such that it is placed in a folder that is created in the directory that contains the results
    -> extracted plots will then be placed in a new folder with the name determined by the constant parameters and are named such that they have an order
'''

def copyFiles(newfilename1, newfilename2, targetdir, resultsfolder, in_plot_name):
    # loop over all files in the target folder and copy them to the resultsfolder, thereby renaming them such that they arrange in order
    print('Copy to folder with name: ', resultsfolder, ' from folder with name: ', targetdir)
    for file in os.listdir(targetdir):
        if not file.endswith(".npz"):                                           # exclude data files
            print('Copy file: ', targetdir+file)
            if file.endswith(".png"):                                           # pick png plots
                if file.startswith("rot_red_PhaseSpace_lastR"):
                    shutil.copy2(targetdir+file, resultsfolder+'/png/lastR/scatt_'+newfilename1+'.png')
                if file.startswith("rot_red_PhaseSpace_meanR"):
                    shutil.copy2(targetdir+file, resultsfolder+'/png/meanR/scatt_'+newfilename2+'.png')
                if file.startswith("imshow_PhaseSpace_lastR"):
                    shutil.copy2(targetdir+file, resultsfolder+'/png/lastR/imsh_'+newfilename1+'.png')
                if file.startswith("imshow_PhaseSpace_meanR"):
                    shutil.copy2(targetdir+file, resultsfolder+'/png/meanR/imsh_'+newfilename2+'.png')
                    # add parameter values to the png file 
                    font = ImageFont.truetype("Calibri.ttf", 256)
                    img = Image.open(resultsfolder+'/png/meanR/imsh_'+newfilename2+'.png')
                    draw = ImageDraw.Draw(img)
                    draw.text((300, 100), in_plot_name, (0, 0, 0), font=font)
                    img.save(resultsfolder+'/png/meanR/imsh_'+newfilename2+'.png')
            else:
                if file.startswith("rot_red_PhaseSpace_lastR"):
                    shutil.copy2(targetdir+file, resultsfolder+'/pdf_lastR/scatt_'+newfilename1+'.pdf')
                if file.startswith("rot_red_PhaseSpace_meanR"):
                    shutil.copy2(targetdir+file, resultsfolder+'/scatt_'+newfilename2+'.pdf')
                if file.startswith("imshow_PhaseSpace_lastR"):
                    shutil.copy2(targetdir+file, resultsfolder+'/pdf_lastR/imsh_'+newfilename1+'.pdf')
                if file.startswith("imshow_PhaseSpace_meanR"):
                    shutil.copy2(targetdir+file, resultsfolder+'/imsh_'+newfilename2+'.pdf')
    return 0

# load parameter data from file, specify delimiter and which line contains the colum description
data = pd.read_csv('DPLLParameters.csv', delimiter="\t", header=2, dtype={'K': np.float, 'Fc': np.float, 'delay': np.float, 'FOmeg': np.float})

# get user input to know which parameter should be analyzed
user_input = raw_input("Please specify parameter {K, Fc, delay} whose dependencies to be analyzed: ")
if user_input == "K":
    # list all possible values of Fc {.,.,.,} and delay {.,.,.,}
    cut_off_freqs = data.Fc.unique()
    cut_off_freqs = cut_off_freqs[~np.isnan(cut_off_freqs)]
    cut_off_freqs.sort()
    delay_values  = data.delay.unique()
    delay_values  = delay_values[~np.isnan(delay_values)]
    delay_values.sort()
    print('\npossible values of Fc = {', cut_off_freqs, '}\n\nand delays = {', delay_values,'}')
    print('analyze K-dependency of basin-stability for fixed values of Fc and delay:')
    Fc    = float(raw_input("for constant cut-off frequency: "))
    delay = float(raw_input("for constant delay-value: "))
    # find all lines which above fixed Parameters and sort them by value
    K_set  = data.loc[(data['delay']==delay) & (data['Fc']==Fc)].sort('K')
    print(K_set)
    # create a folder with name given by the constant parameters
    resultsfolder = 'Kvar-Fc%0.2f-delay%0.2f'%(Fc, delay)
    try:
        os.makedirs(resultsfolder)
        os.makedirs(resultsfolder+'/png/meanR/')
        os.makedirs(resultsfolder+'/png/lastR/')
        os.makedirs(resultsfolder+'/pdf_lastR/')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    # loop trough all lines, extract id and use that to collect the data, i.e., copy all picturefiles into the
    # folder created above and name them by the folder name with 'var' being the variable value from the respective line
    for index, row in K_set.iterrows():
        targetdir = '../%d/results/'%(int(K_set.loc[index, 'id']))              # the directory from which the results are extracted
        newfilename1 = 'K_%0.2f_lR_id%d'%(K_set.loc[index, 'K'], int(K_set.loc[index, 'id'])) # the new file name for the result with the last R
        newfilename2 = 'K_%0.2f_mR_id%d'%(K_set.loc[index, 'K'], int(K_set.loc[index, 'id'])) # the new file name for the result with the mean over R within 2 eigenperiods
        in_plot_name = 'K=%0.2f Hz, Fc=%0.2f Hz, delay=%0.2f s'%(K_set.loc[index, 'K'],K_set.loc[index, 'Fc'],K_set.loc[index, 'delay'])
        #oldfilenames =  listdir('../%d/results/'%(int(K_set.loc[index, 'id'])))
        copyFiles(newfilename1, newfilename2, targetdir, resultsfolder, in_plot_name)

if user_input == "Fc":
    # list all possible values of Fc {.,.,.,} and delay {.,.,.,}
    delay_values  = data.delay.unique()
    delay_values  = delay_values[~np.isnan(delay_values)]
    delay_values.sort()
    coupling_strengths = data.K.unique()
    coupling_strengths = coupling_strengths[~np.isnan(coupling_strengths)]
    coupling_strengths.sort()
    print('possible values of K = {', coupling_strengths, '}\n and delays = {', delay_values,'}')
    print('analyze Fc-dependency of basin-stability for fixed values of K and the delay:')
    K     = float(raw_input("for constant coupling strength: "))
    delay = float(raw_input("for constant delay-value: "))
    # find all lines which above fixed Parameters
    Fc_set  = data.loc[(data['delay']==delay) & (data['K']==K)].sort('Fc')
    # create a folder with name given by the constant parameters
    resultsfolder = 'Fcvar-K%0.2f-delay%0.2f'%(K, delay)
    try:
        os.makedirs(resultsfolder)
        os.makedirs(resultsfolder+'/png/meanR/')
        os.makedirs(resultsfolder+'/png/lastR/')
        os.makedirs(resultsfolder+'/pdf_lastR/')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    # loop trough all lines, extract id and use that to collect the data, i.e., copy all picturefiles into the
    # folder created above and name them by the folder name with 'var' being the variable value from the respective line
    for index, row in Fc_set.iterrows():
        targetdir = '../%d/results/'%(int(Fc_set.loc[index, 'id']))
        newfilename1 = 'Fc_%0.2f_lR_id%d'%(Fc_set.loc[index, 'Fc'], int(Fc_set.loc[index, 'id']))
        newfilename2 = 'Fc_%0.2f_mR_id_%d'%(Fc_set.loc[index, 'Fc'], int(Fc_set.loc[index, 'id']))
        in_plot_name = 'Fc=%0.2f Hz, K=%0.2f Hz, delay=%0.2f s'%(Fc_set.loc[index, 'Fc'],Fc_set.loc[index, 'K'],Fc_set.loc[index, 'delay'])
        #oldfilenames =  listdir('../%d/results/'%(int(K_set.loc[index, 'id'])))
        copyFiles(newfilename1, newfilename2, targetdir, resultsfolder, in_plot_name)

if user_input == "delay":
    # list all possible values of Fc {.,.,.,} and delay {.,.,.,}
    cut_off_freqs = data.Fc.unique()
    cut_off_freqs = cut_off_freqs[~np.isnan(cut_off_freqs)]
    cut_off_freqs.sort()
    coupling_strengths = data.K.unique()
    coupling_strengths = coupling_strengths[~np.isnan(coupling_strengths)]
    coupling_strengths.sort()
    print('possible values of K = {', coupling_strengths, '}\n and Fc = {', cut_off_freqs,'}')
    print('analyze the delay-dependency of basin-stability for fixed values of K and Fc:')
    K  = float(raw_input("for constant coupling strength: "))
    Fc = float(raw_input("for constant cut-off frequency: "))
    # find all lines which above fixed Parameters
    delay_set  = data.loc[(data['K']==K) & (data['Fc']==Fc)].sort('delay')
    # create a folder with name given by the constant parameters
    resultsfolder = 'delayvar-Fc%0.2f-K%0.2f'%(Fc, K)
    try:
        os.makedirs(resultsfolder)
        os.makedirs(resultsfolder+'/png/meanR/')
        os.makedirs(resultsfolder+'/png/lastR/')
        os.makedirs(resultsfolder+'/pdf_lastR/')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    # loop trough all lines, extract id and use that to collect the data, i.e., copy all picturefiles into the
    # folder created above and name them by the folder name with 'var' being the variable value from the respective line
    for index, row in delay_set.iterrows():
        targetdir = '../%d/results/'%(int(delay_set.loc[index, 'id']))
        newfilename1 = 'delay_%0.2f_lR_id%d'%(delay_set.loc[index, 'delay'], int(delay_set.loc[index, 'id']))
        newfilename2 = 'delay_%0.2f_mR_id%d'%(delay_set.loc[index, 'delay'], int(delay_set.loc[index, 'id']))
        in_plot_name = 'delay=%0.2f s, Fc=%0.2f Hz, K=%0.2f Hz'%(delay_set.loc[index, 'delay'],delay_set.loc[index, 'Fc'],delay_set.loc[index, 'K'])
        #oldfilenames =  listdir('../%d/results/'%(int(K_set.loc[index, 'id'])))
        copyFiles(newfilename1, newfilename2, targetdir, resultsfolder, in_plot_name)
