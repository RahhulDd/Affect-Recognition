# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:34:41 2020

@author: dubs
"""
import os
import numpy as np
import pandas as pd

def preprocess():
    data_root = './PACO'
    for phase in ['Train','Test']:
        phase_dir = os.path.join(data_root,phase)
        subject_names = os.listdir(phase_dir)
        
        len_frames = 64 if phase == 'Train' else 128
        
        for subject in subject_names:
            sub_dir = os.path.join(phase_dir,subject)
            motion_names = os.listdir(sub_dir)
            subject = subject.split('_')[0]
            
            for motion in motion_names:
                mot_dir = os.path.join(sub_dir,motion)
                affect_names = os.listdir(mot_dir)
                
                for affect in affect_names:
                    affect_dir = os.path.join(mot_dir,affect)
                    rep_nos = os.listdir(affect_dir)
                    
                    for rep in rep_nos:
                        data_file = os.path.join(affect_dir,rep,subject+'_'+motion+'_'+affect+'_'+rep.split('p')[1]+'_fin.ptd')
                        file = open(data_file,'r')
                        data = file.read()
                        data = data.split('\n')
                        for i in range(0,len(data)):
                            data[i] = data[i].split(" ")
                        df = pd.DataFrame(data=data)
                        df = df.drop(labels=45,axis=1)
                        df = df.drop(labels=[0,len(df)-1],axis=0)
                        df = df.reset_index()
                        del df['index']
                        file.close()
                        df = df.astype(float)
                        animation = df.to_numpy()
                        animation = animation.reshape(15,3,-1)
                        
                        seg_dir = os.path.join(affect_dir,rep,'motions')
                        if not os.path.exists(seg_dir):
                            os.makedirs(seg_dir)
                        
                        total_length = animation.shape[-1]
                        nr_motions = total_length//(len_frames//2)-1
                        for i in range(nr_motions):
                            save_path = os.path.join(seg_dir,'{}.npy'.format(i+1))
                            window = animation[ :,:,i*(len_frames//2):i*(len_frames//2)+len_frames]
                            np.save(save_path,window)
                            print(save_path,window.shape)

if __name__=='__main__':
    preprocess()