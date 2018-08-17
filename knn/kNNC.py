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
from sklearn.neighbors import KNeighborsClassifier
import rtree


#arg[1]=train_path
#arg[2]=dataset_path
#arg[3]=validation_path
#arg[4]=start_k
#arg[5]=end_k
#arg[6]=step_k
#arg[7]=field_class_train
#arg[8]=field_class_validation
#arg[9]=metrics_distance
#arg[10]=weigths_features
#arg[11]=state_applyclass
#arg[12]=assess_text_path
#arg[13]=output_classification

def kNN(path_train,dataset_path,\
                               path_val,start_k,\
                               end_k,step_k,field_class_train,\
                               field_class_val,metric_distance,weigth_feature,\
                               state_applyclass,assess_text_path,classification_path):
        #create text
        f_txt=open(assess_text_path,'w')  
        #Write 
        f_txt.write('Dataset;k;metric;PC;QD;QA'+'\n')
        #get dataframe training samples
        dft=gpd.read_file(path_train)
    
        #get dataframe validation samples
        dfv=gpd.read_file(path_val)
        
        #get names data set
        dataset_names=[f for f in os.listdir(dataset_path)  if f.endswith('.shp')]
        #best parameters
        best_parameters={'k':0,'metric':0.,'dataset':None}
        #acurcia
        acuracia=0.0
        #segmentations file
        for seg in dataset_names:                
            #Selecionar arquivos .shp          
            #f_txt.write(segs_path+os.sep+seg+'\n')
            print (dataset_path+os.sep+seg)
            #Ler segmentacoes
            dfs=gpd.read_file(dataset_path+os.sep+seg)
            #create validation samples merge attribute spatial join
            dfjv=gpd.sjoin(dfv,dfs,how="inner", op='intersects')
            #Criar amostras de treinamento, merge attribute spatial join
            dfjt=gpd.sjoin(dft,dfs,how="inner", op='intersects')
            
            #Get features and remove geometry and id_seg
            dfs.drop(['geometry','id_seg'],axis=1,inplace =True)
            features=dfs.columns
            #Assess metric distance
            if metric_distance == 'All':
                metric=[1,2,3]
            elif metric_distance == 'Manhattan (p=1)':
                metric=[1]
            elif metric_distance =='Euclidean (p=2)':
                metric=[2]
            else:
                metric=[3]
       
            #Avaliar parametros da segmentacao
            for k_v in range(int(start_k),int(end_k),int(step_k)):
                for p_v in metric:
                    #criar modelo Random Forest
                    clf = KNeighborsClassifier( n_neighbors=k_v,p=p_v,weights=weigth_feature)
                    #Ajustar modelo
                    modelTree = clf.fit(dfjt[features].values, dfjt[field_class_train])
                    #Classificar
                    clas = modelTree.predict(dfjv[features].values)
                   
                    #Calculate PC
                    pc,qd,qa,matrix=pontius2011(dfjv[field_class_val],clas)
                    print (pc,qd,qa)
                    f_txt.write(seg+';'+str(k_v)+';'+ str(p_v)+';'+str(round(pc,4))+';'+str(round(qd,4))+';'+str(round(qa,4))+'\n') 
                    #Avaliar a acuracia
                    
                    if pc > acuracia:
                        acuracia=pc
                        #Guardar parametros random forest
                        
                        best_parameters['k']=k_v
                        best_parameters['metric']=p_v
                        best_parameters['dataset']=seg
        del(dfs,dfjv,dfjt)           
        #classificar segmentacao
        f_txt.write('############# Best Parameters #############'+'\n')
        f_txt.write('dataset: '+best_parameters['dataset']+' - '+'k: '+str(best_parameters['k'])+ ' - metric:'+str(best_parameters['metric'])+'\n')
        ###################### classify best case##############################
        if bool(state_applyclass) :
            #Ler segmentacoes
            df_dataset=gpd.read_file(dataset_path+os.sep+best_parameters['dataset'])
            #create validation samples merge attribute spatial join
            dfjv=gpd.sjoin(dfv,df_dataset,how="inner", op='intersects')
            #Criar amostras de treinamento, merge attribute spatial join
            dfjt=gpd.sjoin(dft,df_dataset,how="inner", op='intersects')
            #criar modelo KNN
            clf = KNeighborsClassifier( n_neighbors=best_parameters['k'],p=best_parameters['metric'],weights=weigth_feature)
            #Ajustar modelo
            model = clf.fit(dfjt[features].values, dfjt[field_class_train])
            #Classificar
            clas = model.predict(dfjv[features].values)                      
            #Calculate PC
            pc,qd,qa,matrix=pontius2011(dfjv[field_class_val],clas)
            #Classificar
            classification = modelTree.predict(df_dataset[features].values)
            ##create aux DF classification
            df_dataset['classes']=classification
            #output classification
            df_dataset[['geometry','classes']].to_file( classification_path)
 
            f_txt.write('############# Confusion Matrix #############'+'\n')
            f_txt.write(str(matrix)+'\n')
            #del
            del(df_dataset,dfjv,dfjt,dfv,dft)
        else:
            pass
       
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
        
def pontius2011(labels_validation,classifier):
        #get class
        labels = np.unique(labels_validation)
        #Get total class
        n_labels=labels.size        
        print ('n labels: ',n_labels)
        #create matrix 
        sample_matrix = np.zeros((n_labels,n_labels))
        #print sample_matrix
        #print np.count_nonzero(classifier==labels_validation)

        #Loop about labels
        for i,l in enumerate(labels):
            #Assess label in classifier
            selec=classifier==l
            print ( selec.any())
            if selec.any():
                #Get freqs
                coords,freqs=np.unique(labels_validation[selec],return_counts=True)
                print (coords,freqs)
                #insert sample_matrix
                sample_matrix[i,coords-1]=freqs
                print( 'l, Freqs: ',l,'---',freqs)
            
        print (sample_matrix)
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
        sample_total = sample_total.reshape(n_labels,1)
        #Population total: Image classification or labels validation (random)
        population = np.bincount(labels_validation)
        #Remove zero
        population = population[1:]
        print (population)
        
        #population matrix
        pop_matrix = np.multiply(np.divide(sample_matrix,sample_total,where=sample_total!=0),(population.astype(float)/population.sum()))
        
        #comparison total: Sum rows pop_matrix
        comparison_total = np.sum(pop_matrix, axis=1)
        #reference total: Sum columns pop_matrix
        reference_total = np.sum(pop_matrix, axis=0)
        #overall quantity disagreement
        quantity_disagreement=(abs(reference_total-comparison_total).sum())/2.
        #overall allocation disagreemen
        dig =pop_matrix.diagonal()
        comp_ref=np.dstack((comparison_total-dig,reference_total-dig))
        #allocation_disagreemen=((2*np.min(comp_ref,-1)).sum())/2.
        #proportion correct
        proportion_correct = np.trace(pop_matrix)
        allocation_disagreemen=1-(proportion_correct+quantity_disagreement)
        #print 'Quantity disagreement :',quantity_disagreement
        #print 'Allocation disagreemen :',allocation_disagreemen
        #print 'Proportion correct: ',proportion_correct
        print ('PC: ',proportion_correct, ' DQ: ',quantity_disagreement, 'AD: ',allocation_disagreemen)
        return proportion_correct, quantity_disagreement, allocation_disagreemen,sample_matrix
    

if __name__ == '__main__':
    
    #Run function
    kNN(sys.argv[1],sys.argv[2],\
                           sys.argv[3],sys.argv[4],\
                           sys.argv[5],sys.argv[6],sys.argv[7],\
                           sys.argv[8],sys.argv[9],sys.argv[10],\
                           sys.argv[11],sys.argv[12],sys.argv[13])
    print ("Finalizou k-NN")


