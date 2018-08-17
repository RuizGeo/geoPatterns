#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:18:30 2018

@author: ruiz
"""
import geopandas as gpd
import numpy as np
import os
import sys
import time
from sklearn import ensemble
import rtree
#arg[1]=path_train
#arg[2]=segs_path
#arg[3]=path_val
#arg[4]=start_est
#arg[5]=end_est
#arg[6]=step_est
#arg[7]=start_dp
#arg[8]=end_dp
#arg[9]=step_dp
#arg[10]=field_class_train
#arg[11]=field_class_val
#arg[12]=criterion_split
def RandomForestClassifier(path_train,segs_path,\
                           path_val,start_est,\
                           end_est,step_est,start_dp,\
                           end_dp,step_dp,field_class_train,\
                           field_class_val,criterion_split):
    #create text
    f_txt=open(segs_path+os.sep+'assessment_RFC.txt','w')  
    #get dataframe training samples
    dft=gpd.read_file(path_train)

    #get dataframe validation samples
    dfv=gpd.read_file(path_val)
    
    #get names segmentations
    segs_names=[f for f in os.listdir(segs_path)  if f.endswith('.shp')]
    #segmentations file
    for seg in segs_names:                
        #Selecionar arquivos .shp          
        f_txt.write(segs_path+os.sep+seg+'\n')
        print (segs_path+os.sep+seg)
        #Ler segmentacoes
        dfs=gpd.read_file(segs_path+os.sep+seg)
        #create validation samples merge attribute spatial join
        dfjv=gpd.sjoin(dfv,dfs,how="inner", op='intersects')
        #Criar amostras de treinamento, merge attribute spatial join
        dfjt=gpd.sjoin(dft,dfs,how="inner", op='intersects')
        
        #Get features and remove geometry and id_seg
        dfs.drop(['geometry','id_seg'],axis=1,inplace =True)
        #Get columns names (Fields)
        features=dfs.columns
        try:
            dfjt[features]
            dfjv[features]
        except:
            print ("Error features names: ",features)
            return 'features'
            
        #acurcia
        acuracia=0    
        best_parameters=[]        
        #Avaliar parametros da segmentacao
        for t in range(int(start_est),int(end_est),int(step_est)):
            for md in range(int(start_dp),int(end_dp),int(step_dp)):
                #criar modelo Random Forest
                clf = ensemble.RandomForestClassifier( n_estimators =t, max_depth =md)
                #Ajustar modelo
                modelTree = clf.fit(dfjt[features].values, dfjt[field_class_train])
                #Classificar
                clas = modelTree.predict(dfjv[features].values)
                #Calc accuracy
                acuracia=np.count_nonzero(clas==dfjv[field_class_val])
                f_txt.write('trees: '+str(t)+' - '+'Depth: '+ str(md)+' - '+'acc: '+str(acuracia)+'\n') 
                #Avaliar a acuracia
                if np.count_nonzero(clas==dfv[field_class_train]) > acuracia:
                    acuracia=np.count_nonzero(clas==dfjv[field_class_val])                   
                    #Guardar parametros random forest
                    
                    best_parameters.append([t,md])
                   
       #classificar segmentacao
        #obter os melhores parametros
        #trees, max_d=
        print (best_parameters)
        #criar o modelo Random Forest
        #clf = ensemble.RandomForestClassifier( n_estimators =trees, max_depth =max_d,criterion=criterion_split)
        #Ajustar o modelo
        #modelTree = clf.fit(dfj[features].values, dfj[field_class_train])
        #Classificar segmentacao
        #dfs['classify']=modelTree.predict(dfs.values)
        #Calcular a probabilidade e converter para String
        #probs=[str(row) for row in np.round(modelTree.predict_proba(dfs[features].values)*100,2).tolist()]
        #insert geodata frame
        #dfs['probs']=probs
        #Save classify        
        #dfs[['geometry','classify','probs']].to_file(segs_path+os.sep+'class_'+seg)
        #Criar datafame com 
    f_txt.close()
    
def pontius2011(labels_validation,classifier, n_labels):
    #create matrix 
    sample_matrix = np.zeros((n_labels,n_labels))
    #print sample_matrix
    #print np.count_nonzero(classifier==labels_validation)
    #get class
    labels = np.unique(classifier)
    #Loop about labels
    for i,l in enumerate(labels):
        #Get freqs
        coords,freqs=np.unique(labels_validation[classifier==l],return_counts=True)
        print( 'l, Freqs: ',l,'---',freqs)
        
        sample_matrix[i,coords-1]=freqs
    #print sample_matrix
    #Sample matrix: samples vs classification
    #sample_matrix=np.histogram2d(classifier,labels_validation,bins=(n_labels,n_labels))[0]
    #coo =np.array([4,5,8,9,11,12,13])-1
    #sample_matrix=sample_matrix[:,coo]
    #sample_matrix=sample_matrix[coo,:]
    print (sample_matrix.shape)
    #Sum rows sample matrix
    sample_total = np.sum(sample_matrix, axis=1)
    print ('sum rows: ',sample_total)
    #reshape sample total
    sample_total = sample_total.reshape(7,1)
    #Population total: Image classification or labels validation (random)
    population = np.bincount(labels_validation)
    #Remove zero
    population = population[1:]

    #population matrix
    pop_matrix = np.multiply(np.divide(sample_matrix,sample_total),(population.astype(float)/population.sum()))
    #comparison total: Sum rows pop_matrix
    comparison_total = np.sum(pop_matrix, axis=1)
    #reference total: Sum columns pop_matrix
    reference_total = np.sum(pop_matrix, axis=0)
    #overall quantity disagreement
    quantity_disagreement=(abs(reference_total-comparison_total).sum())/2.
    #overall allocation disagreemen
    dig =pop_matrix.diagonal()
    comp_ref=np.dstack((comparison_total-dig,reference_total-dig))
    allocation_disagreemen=((2*np.min(comp_ref,-1)).sum())/2.
    #proportion correct
    proportion_correct = np.trace(pop_matrix)
    
    #print 'Quantity disagreement :',quantity_disagreement
    #print 'Allocation disagreemen :',allocation_disagreemen
    #print 'Proportion correct: ',proportion_correct
    print ('PC: ',proportion_correct, ' DQ: ',quantity_disagreement, 'AD: ',allocation_disagreemen)
    return proportion_correct, quantity_disagreement, allocation_disagreemen
        
if __name__ == '__main__':
    
    #Run function
    RandomForestClassifier(sys.argv[1],sys.argv[2],\
                           sys.argv[3],sys.argv[4],\
                           sys.argv[5],sys.argv[6],sys.argv[7],\
                           sys.argv[8],sys.argv[9],sys.argv[10],\
                           sys.argv[11],sys.argv[12])

