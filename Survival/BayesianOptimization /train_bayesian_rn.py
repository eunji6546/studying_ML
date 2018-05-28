from keras.layers import Dense, Dropout, Input
from keras.models import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import ModelCheckpoint
import os
import argparse
import pickle
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn.decomposition import PCA 
import pickle
from bayes_opt import BayesianOptimization
import 
from sklearn.decomposition import PCA 
import pandas as pd
import rn_ae
import variables

# RN version 
def cindex(cen, surv, y_pred):
    N = surv.shape[0]
    Comparable = np.zeros([N,N])

    for i in range(N):
        for j in range(N):
            if cen[i] == 0 and cen[j] == 0:
                if surv[i] != surv[j]:
                    Comparable[i, j] = 1

            elif cen[i] == 1 and cen[j] == 1:
                Comparable[i, j] = 0

            else: # one sample is censored and the other is not
                if cen[i] == 1:
                    if surv[i] >= surv[j]:
                        Comparable[i, j] = 1
                    else:
                        Comparable[i, j] = 0
                else: # cen[j] == 1
                    if surv[j] >= surv[i]:
                        Comparable[i, j] = 1
                    else:
                        Comparable[i, j] = 0

    p2, p1 = np.where(Comparable==1)
    Y = y_pred

    c=0
    N_valid_sample = p1.shape[0]
    for i in range(N_valid_sample):
        if cen[p1[i]] == 0 and cen[p2[i]] == 0:
            if Y[p1[i]] == Y[p2[i]]:
                c = c + 0.5
            elif Y[p1[i]] > Y[p2[i]] and surv[p1[i]] > surv[p2[i]]:
                c = c + 1
            elif Y[p2[i]] > Y[p1[i]] and surv[p2[i]] > surv[p1[i]]:
                c = c + 1

        elif cen[p1[i]] == 1 and cen[p2[i]] == 1:
            continue # do nothing - samples cannot be ordered

        else: # one sample is censored and the other is not
            if cen[p1[i]] == 1:
                if Y[p1[i]] > Y[p2[i]] and surv[p1[i]] > surv[p2[i]]:
                    c = c + 1
                elif Y[p1[i]] == Y[p2[i]]:
                    c = c + 0.5

            else: # cen[p2[i]] == 1
                if Y[p2[i]] > Y[p1[i]] and surv[p2[i]] > surv[p1[i]]:
                    c = c + 1
                elif Y[p1[i]] == Y[p2[i]]:
                    c = c + 0.5

    c = c*1.0 / N_valid_sample
    return c

def run_rn(g_mlp_layers_0, f_mlp_layers_0,rn_dim ):
  
#(g_mlp_layers_0, g_mlp_layers_1, f_mlp_layers_0, f_mlp_layers_1,rn_dim ):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "3" #gpunum

    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ktf.set_session(get_session())

    # Hyper Parameters 
    g_mlp_layers_0 = int(g_mlp_layers_0) * 16 * 2
    g_mlp_layers_1 = int(g_mlp_layers_0) #* 16
    f_mlp_layers_0 = int(f_mlp_layers_0) * 32 * 2
    f_mlp_layers_1 = int(f_mlp_layers_0) #* 32
    rn_dim = int(rn_dim)

    variables.rn_dim = rn_dim
    g_MLP_layers = [g_mlp_layers_0, g_mlp_layers_1]
    f_MLP_layers = [f_mlp_layers_0, f_mlp_layers_1]
    
    rn_args = {
        "g_MLP_layers": g_MLP_layers,
        "f_MLP_layers": f_MLP_layers,
        "learning_rate": variables.rn_learning_rate,
        "epochs": variables.rn_epochs,
        "batch_size": variables.rn_batch_size,
        "model_name": "rn_dim%d_qry%s_by%s" %(variables.rn_dim, str(variables.rn_query), str(variables.rn_relation_by))
    }
    print(rn_args)
    # Load Data 
    cancer_type_list = ['LUAD', 'LUSC'] 

    for cancer_type in cancer_type_list: 

        # TODO : do i need to change this to K-fold?? 

        rn_data_all = rn_ae.load_data_by_sample(cancer_type)
        X = rn_data_all.drop(['y','c'], axis=1).values
        print "X.shape ", X.shape
        Y = rn_data_all['y'].values
        print "Y.shape ", Y.shape
        C = rn_data_all['c'].values
        print "C.shape ", C.shape

        x_trn, x_tst, y_trn, y_tst, c_trn, c_tst = \
        train_test_split(X, Y, C, test_size=80, random_state=7)

        model = rn_ae.train_RNs(x_trn, y_trn, rn_args)

        
        rn_predict = model.predict(x_tst)
        rn_score = cindex(c_tst, y_tst, rn_predict)

        print "[", cancer_type,"]"
        print rn_args
        print rn_score
        return rn_score 


if __name__ == "__main__":
    
    bo_dict = { 
        "g_mlp_layers_0" : (1, 3),
        #"g_mlp_layers_1" 
        "f_mlp_layers_0" : (1, 3),
       # "f_mlp_layers_1"
        "rn_dim" : (500, 1500)
    }

    v1BO = BayesianOptimization(run_rn, bo_dict,verbose=True)
    
    v1BO.explore({
        "g_mlp_layers_0" : (1,1, 3),
        #"g_mlp_layers_1" 
        "f_mlp_layers_0" : (1,1, 3),
       # "f_mlp_layers_1"
        "rn_dim" : (500, 500, 1500)
    })
 
    gp_params = {"alpha": 1e-5}
    
    v1BO.maximize(init_points = 2, n_iter=30, acq='ucb', kappa=5)
    
    print('Final Results')
    #print('max %f' % v1BO.res['max']['max_val'])
    #print('***<max>****')
    #print(v1BO.res['max'])
    #print('***<all>***')
    #print(v1BO.res['all'])
    results.append(v1BO.res['all'])
    #print(results)
    print(v1BO.res)

    with open('./rn/BO_Result_'+cancer_type+'.txt','at' ) as f:
        
        params =v1BO.res['all']['params']
        values = v1BO.res['all']['values']
        keys = params[0].keys() 

        for i in range(2) : 
            line = [cancer_type, feature_type] 
          
            for k in keys : 
                line.append(str(params[i][k]))
            line.append(str(values[i]))
            f.write('\t'.join(line)+'\n')
     
