#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 10:44:46 2018

@author: ruiz
"""






def importLibs():
    
    from sklearn import ensemble
    import geopandas as gpd
    import numpy as np
    import os
    import sys
    import time
    #from sklearn import ensemble
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
                               field_class_val,criterion_split,path_assess_file,stateCheckBox,classification_path):
    
        
        #create text
        f_txt=open(path_assess_file,'w')  
        #Write 
        f_txt.write('Seg;Trees;Depth;PC;QD;QA'+'\n')
        #get dataframe training samples
        dft=gpd.read_file(path_train)
    
        #get dataframe validation samples
        dfv=gpd.read_file(path_val)
        
        #get names segmentations
        segs_names=[f for f in os.listdir(segs_path)  if f.endswith('.shp')]
        #best parameters
        best_parameters={'Trees':0,'Depth':0}
        #acurcia
        acuracia=0.0
        #segmentations file
        for seg in segs_names:                
            #Selecionar arquivos .shp          
            #f_txt.write(segs_path+os.sep+seg+'\n')
            print (segs_path+os.sep+seg)
            #Ler segmentacoes
            dfs=gpd.read_file(segs_path+os.sep+seg)
            #create validation samples merge attribute spatial join
            dfjv=gpd.sjoin(dfv,dfs,how="inner", op='intersects')
            #Criar amostras de treinamento, merge attribute spatial join
            dfjt=gpd.sjoin(dft,dfs,how="inner", op='intersects')
            
            #Get features and remove geometry and id_seg
            dfs=dfs.drop(columns=['geometry','id_seg'])
            features=dfs.columns


            #Avaliar parametros da segmentacao
            for t in range(int(start_est),int(end_est)+int(step_est),int(step_est)):
                for md in range(int(start_dp),int(end_dp)+int(step_dp),int(step_dp)):
                    #criar modelo Random Forest
                    clf = ensemble.RandomForestClassifier( n_estimators =t, max_depth =md)
                    #Ajustar modelo
                    modelTree = clf.fit(dfjt[features].values, dfjt[field_class_train])
                    #Classificar
                    clas = modelTree.predict(dfjv[features].values)
                    
                    #Calculate PC
                    pc,qd,qa=pontius2011(dfjv[field_class_val],clas)
                    print (pc,qd,qa)
                    f_txt.write(seg+';'+str(t)+';'+ str(md)+';'+str(round(pc,4))+';'+str(round(qd,4))+';'+str(round(qa,4))+'\n') 
                    #Avaliar a acuracia
                    print('Acc: '+str(acuracia)+' Pc: '+str(pc))
                    if pc > acuracia:
                        acuracia=pc
                        #Guardar parametros random forest
                        
                        best_parameters['Trees']=t
                        best_parameters['Depth']=md
                        best_parameters['Seg']=seg
        del(dfs,dfjv,dfjt)           
        #classificar segmentacao
        f_txt.write('############# Best Parameters #############'+'\n')
        f_txt.write('Seg: '+best_parameters['Seg']+' - '+'Trees: '+str(best_parameters['Trees'])+ ' - Depth:'+str(best_parameters['Depth'])+'\n')
        ###################### classify best case##############################
        if bool(stateCheckBox) :
            #Ler segmentacoes
            df_dataset=gpd.read_file(segs_path+os.sep+best_parameters['Seg'])
            #create validation samples merge attribute spatial join
            dfjv=gpd.sjoin(dfv,df_dataset,how="inner", op='intersects')
            #Criar amostras de treinamento, merge attribute spatial join
            dfjt=gpd.sjoin(dft,df_dataset,how="inner", op='intersects')
            #criar modelo Random Forest
            clf = ensemble.RandomForestClassifier( n_estimators =best_parameters['Trees'], max_depth =best_parameters['Depth'])
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
            f_txt.write('############# Features #############'+'\n')
            f_txt.write(str(features)+'\n')
            f_txt.write('############# Features Importances #############'+'\n')
            f_txt.write(str(model.feature_importances_)+'\n')  
            f_txt.write('############# Confusion Matrix #############'+'\n')
            f_txt.write(str(matrix)+'\n')
            #del
            del(df_dataset,dfjv,dfjt,dfv,dft)
        else:
            pass
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
        return proportion_correct, quantity_disagreement, allocation_disagreemen
    


