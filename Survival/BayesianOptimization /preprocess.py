from pandas import DataFrame

import time
import os
import pandas as pd
import numpy as np
import pickle

# VERSIOn : just concat 

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):

        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)

def one_feature_df(cancer_type, feature_type, overlap_patients ):

    print(cancer_type, feature_type)
    with Timer('processing..'):
        filename = '../test_bong/data/002common_%s_%s_%s.txt' % (cancer_type, 'T', feature_type)
        if not (os.path.exists(filename)):
            print("file not exists")
            return 
        df_tumor = DataFrame.from_csv(filename, sep='\t')
        df_tumor = df_tumor.transpose()
        print df_tumor.shape
      
        filename = '../test_bong/data/002common_%s_%s_%s.txt' % (cancer_type, 'N', feature_type)
        df_normal = pd.read_csv(filename, sep='\t',index_col=False)
      
        #if ( feature_type.startswith("beta") ) :
        #    df_normal = df_normal[df_normal.columns.tolist()[1:]] 
        df_normal.index = df_normal.iloc[:,0]
        
        df_normal = df_normal[df_normal.columns.tolist()[1:]] 
        
            
        df_normal = df_normal.transpose()
       
        

        df_normal.columns = [name.split('_')[0] for name in df_normal.columns.values]
        print df_normal.shape
        df_all = df_tumor.append(df_normal)

        print df_all.shape

        df_pid = DataFrame.from_csv('../data/filtered_pid.txt', sep='\t')

        filtered_pid_list = df_pid.index.values
      

        df_all.index = map(lambda index: index.split('|')[-1][:12], df_all.index)

        for pid in df_all.index.values:
            # if pid.split('|')[-1][:12] not in filtered_pid_list:
            if pid not in filtered_pid_list or pid not in overlap_patients:
                # print 'drop %s' % pid
                df_all = df_all.drop(pid)

        print df_all.shape

    
        return df_all 

def just_concat(cancer_type, feature_type_list) :

    with open('./overlap_sampleID_%s.txt' % cancer_type, 'r')as f:
        overlap_patients = f.read()
    overlap_patients = overlap_patients.split('\t')
    print(overlap_patients)

    df1 = one_feature_df(cancer_type, feature_type_list[0],overlap_patients)
    df2 = one_feature_df(cancer_type, feature_type_list[1],overlap_patients)
    df3 = one_feature_df(cancer_type, feature_type_list[2],overlap_patients)
    
    df1.columns = [x+"_methyl" for x in df1.columns]
    df2.columns = [x+"_rna" for x in df2.columns]
    df3.columns = [x+"_mirna" for x in df3.columns] 
    print(df1.index)
    print(df2.index)

    print(df1.columns)
    print(df2.columns)
    df_all = pd.merge(pd.merge(df1, df2, right_index=True, left_index=True)
        , df3, right_index=True, left_index=True)
    df_all['index'] = df_all.index

    df_all =df_all.drop_duplicates('index', 'first')
    df_all = df_all.iloc[:,:-1]
    print(df_all.index)
    print(df_all.columns)
    print(df_all.shape)

    df_all["censor"] = ""
    df_all["survival"] = ""




    with Timer('imputation..'):
        df_all_imputed = df_all.fillna(df_all.mean())

    with Timer('drop -1 features..'):
        cols = list(df_all_imputed)
        nunique = df_all_imputed.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        df_all_imputed = df_all_imputed.drop(cols_to_drop, axis=1)
        print df_all_imputed.values.min(), df_all_imputed.values.max()

    with Timer('normalization..'):
        df_all_imputed_normalized = (df_all_imputed - df_all_imputed.mean()) / (df_all_imputed.max() - df_all_imputed.min())

        # print df_all_imputed.mean(), df_all_imputed.max(), df_all_imputed.min()
        print df_all_imputed_normalized.values.min(), df_all_imputed_normalized.values.max()
        print df_all_imputed_normalized.shape

    with Timer('join..'):
        df_pid = DataFrame.from_csv('../data/filtered_pid.txt', sep='\t')
        df_xy = df_all_imputed_normalized.join(df_pid)

    print df_xy.index.values.shape
    print df_xy.values[:, :-2].shape
    print df_xy.values[:, -2:].shape

    with Timer('save..'):
        with open('../test_bong/data/overlap_sd15_%s.pkl' % (cancer_type), 'wb') as handle:
            pickle.dump(df_xy.index.values, handle) # labels
            pickle.dump(df_xy.values[:, :-2], handle) # X (20166)
            pickle.dump(df_xy.values[:, -2:].astype(int), handle) # Y (censored, survival) 







def run(cancer_type, feature_type):
    print cancer_type, feature_type
    if os.path.exists('../data/%s.%s.pkl' % (cancer_type, feature_type)) : 
        print 'already preprocessed'
        return 

    with Timer('processing..'):
        filename = '../data/002common_%s_%s_%s.txt' % (cancer_type, 'T', feature_type)
        if not (os.path.exists(filename)):
            print("file not exists")
            return 
        df_tumor = DataFrame.from_csv(filename, sep='\t')
        df_tumor = df_tumor.transpose()
        print df_tumor.shape

        #if (feature_type.startswith("beta") ) :
#            filename = '../data/002common_%s_%s_%s.txt' % (cancer_type, 'N', feature_type)
#            df_normal = pd.read_csv(filename, sep='\t',index_col=False)
#            df_normal.drop(['Unnamed: 0'], axis=1, inplace=True)
#            df_normal.to_csv(filename, sep='\t', index=False) 
 
        filename = '../data/002common_%s_%s_%s.txt' % (cancer_type, 'N', feature_type)
        df_normal = pd.read_csv(filename, sep='\t',index_col=False)
        print df_normal.columns
        if (feature_type.startswith("beta") ) :
            df_normal = df_normal[df_normal.columns.tolist()[1:]]
            df_normal.index = df_normal.iloc[:,0]
            
            print(df_normal.columns)
            
        df_normal = df_normal.transpose()
        print(df_normal.head())
        

        df_normal.columns = [name.split('_')[0] for name in df_normal.columns.values]
        print df_normal.shape
        df_all = df_tumor.append(df_normal)

        print df_all.shape

        df_pid = DataFrame.from_csv('../data/filtered_pid.txt', sep='\t')

        filtered_pid_list = df_pid.index.values
        df_all["censor"] = ""
        df_all["survival"] = ""

        df_all.index = map(lambda index: index.split('|')[-1][:12], df_all.index)

        for pid in df_all.index.values:
            # if pid.split('|')[-1][:12] not in filtered_pid_list:
            if pid not in filtered_pid_list:
                # print 'drop %s' % pid
                df_all = df_all.drop(pid)


        print df_all.shape


    with Timer('imputation..'):
        df_all_imputed = df_all.fillna(df_all.mean())

    with Timer('drop -1 features..'):
        cols = list(df_all_imputed)
        nunique = df_all_imputed.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        df_all_imputed = df_all_imputed.drop(cols_to_drop, axis=1)
        print df_all_imputed.values.min(), df_all_imputed.values.max()

    with Timer('normalization..'):
        df_all_imputed_normalized = (df_all_imputed - df_all_imputed.mean()) / (df_all_imputed.max() - df_all_imputed.min())

        # print df_all_imputed.mean(), df_all_imputed.max(), df_all_imputed.min()
        print df_all_imputed_normalized.values.min(), df_all_imputed_normalized.values.max()
        print df_all_imputed_normalized.shape

    with Timer('join..'):
        df_xy = df_all_imputed_normalized.join(df_pid)

    print df_xy.index.values.shape
    print df_xy.values[:, :-2].shape
    print df_xy.values[:, -2:].shape

    with Timer('save..'):
        with open('../data/%s.%s.pkl' % (cancer_type, feature_type), 'wb') as handle:
            pickle.dump(df_xy.index.values, handle) # labels
            pickle.dump(df_xy.values[:, :-2], handle) # X (20166)
            pickle.dump(df_xy.values[:, -2:].astype(int), handle) # Y (censored, survival)



if __name__=='__main__':
    cancer_type_list = ['LUAD', 'LUSC']
    feature_type_list =  ['betaValue_methyl.hg19.sd15', 'normalized_RNA-seq.hg19', 'RPM_miRNA.hg19.mirbase20'] #, 'betaValue_methyl.hg19']
    for cancer_type in cancer_type_list:
        #for feature_type in feature_type_list:
        #    run(cancer_type, feature_type)
        just_concat(cancer_type, feature_type_list )