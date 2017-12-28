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
import AE_again_read
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
    g_mlp_layers_1 = int(g_mlp_layers_0) * 16
    f_mlp_layers_0 = int(f_mlp_layers_0) * 32 * 2
    f_mlp_layers_1 = int(f_mlp_layers_0) * 32
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
     
    
    """
    cancer_type_list = ['LUAD', 'LUSC']
    feature_type_list =  ['betaValue_methyl.hg19.sd15', 'normalized_RNA-seq.hg19', 'RPM_miRNA.hg19.mirbase20'] #, 'betaValue_methyl.hg19']
    
    for cancer_type in cancer_type_list : 
        for sd in [15,17,20]:
            with open('../test_bong/data/overlap_sd%d_%s.pkl' % (sd,cancer_type), 'rb') as handle:
                print cancer_type, sd 
                labels = pickle.load(handle)
                x = pickle.load(handle)
                y = pickle.load(handle)
                
                print(x.shape)
    """
                

"""




def my_cindex(cen, surv):
    def cindex(y_true, y_pred):
        #print(y_true)
        #print(y_pred)
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
        return 1-c 
    return cindex

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

class MyCallback(ModelCheckpoint):
    def __init__(self, res_list, filepath, data, real_save=True, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, cancer_type='NULL', feature_type='Not', 
                 thr = 0, dropout_prob = 0.8, dimension=400, activate="relu",
                 AE1=800, AE2=400):
        super(MyCallback, self).__init__(filepath, monitor, verbose,
                                         save_best_only, save_weights_only,
                                         mode, period)

        self.x_trn, self.c_trn, self.s_trn, \
        self.x_dev, self.c_dev, self.s_dev, \
        self.x_tst, self.c_tst, self.s_tst = data

        self.score_trn = 0
        self.score_dev = 0
        self.score_tst = 0
        self.real_save = real_save
        self.cancer_type = cancer_type
        self.feature_type = feature_type
        self.threshold = thr
        self.dropout_prob = dropout_prob
        self.dim = dimension
        self.mode = activate
        self.ae1 = AE1
        self.ae2 = AE2
        self.res_list = res_list 

    def evaluate(self):
        pred_trn = self.model.predict(self.x_trn)
        cindex_trn = cindex(self.c_trn, self.s_trn, pred_trn)

        pred_dev = self.model.predict(self.x_dev)
        cindex_dev = cindex(self.c_dev, self.s_dev, pred_dev)

        pred_tst = self.model.predict(self.x_tst)
        cindex_tst = cindex(self.c_tst, self.s_tst, pred_tst)

        if self.score_dev < cindex_dev:
            self.score_trn = cindex_trn
            self.score_dev = cindex_dev
            self.score_tst = cindex_tst

            print("\nupdated!!")
            if self.real_save == True:
                if self.save_weights_only:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)

        print '\n[This Epoch]'
        print '\t'.join(map(str, [cindex_trn, cindex_dev, cindex_tst]))
        print '[Current Best]'
        print '\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst]))

    def on_train_end(self, logs=None):
        print '[Best:on_train_end]'
        print '\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst]))
        with open('./Result_BO/result_'+self.mode+'_pca'+str(self.threshold)
        +'drp'+str(self.dropout_prob)+'_dim'+str(self.dim)+'AE'+str(self.ae1)+':'+str(self.ae2) + '.txt', 'at') as f : 
            f.write("%s\t%s\t%s\t%s\t%s\n" %(self.cancer_type, self.feature_type, str(self.score_trn), str(self.score_dev), str(self.score_tst) ))
        self.res_list.append("%s\t%s\t%s\t%s\t%s\n" %(self.cancer_type, self.feature_type, str(self.score_trn), str(self.score_dev), str(self.score_tst)))

    def on_epoch_end(self, epoch, logs=None):
        self.evaluate()

def run(gpunum, cancer_type, feature_type, attempt):

    batch_size = 32
    epochs = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum
    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ktf.set_session(get_session())



    results = [] 

    def scoreofModel(cancer_type, feature_type, attempt):

        def inner_SoM( pca, dropout, hidden_dims, ae_dim1, ae_dim2):
            print("**scoreofModel pca " + str(pca) + " dropout "+str(dropout) + 
                " hidden dims " + str(hidden_dims) + " dim1 " + str(ae_dim1) + " dim2 "+str(ae_dim2))
            print("ct %s ft %s attempt %d" %(cancer_type, feature_type, attempt))

            hidden_dims = int(hidden_dims)
            ae_dim1 = int(ae_dim1)
            ae_dim2 = int(ae_dim2)
        
            # AE 
            with open('../test_bong/data/overlap_%s.pkl' % (cancer_type), 'rb') as handle:
                labels = pickle.load(handle)
                x = pickle.load(handle)
                y = pickle.load(handle)
            x_trn, x_tst, c_trn, c_tst, s_trn, s_tst, l_trn, l_tst = \
                train_test_split(x, y[:, 0], y[:, 1], labels, test_size=80, random_state=7)
            x_trn, x_tst = AE_again_read.AE_model_save(cancer_type,feature_type, ae_dim1,ae_dim2,x_trn,x_tst)
            clf = PCA(pca,whiten=True)     
            x_trn=clf.fit_transform(x_trn)
            x_tst=clf.transform(x_tst)    

            x_trn, x_dev, c_trn, c_dev, s_trn, s_dev, l_trn, l_dev = train_test_split(x_trn, c_trn, s_trn, l_trn, test_size=20, random_state=7)
            data = tuple((x_trn, c_trn, s_trn, x_dev, c_dev, s_dev, x_tst, c_tst, s_tst))
   
    
            def ModelV1(model_input) : 
                z = Dropout(dropout)(model_input)
                z = Dense(hidden_dims, activation='relu')(z)
                z = Dropout(dropout)(z)
                z = Dense(hidden_dims, activation='relu')(z)
                model_output = Dense(1, activation=None)(z)
                model = Model(model_input, model_output)
                #model.compile(loss=my_cindex(c_tst, s_tst), optimizer='adam')#,metrics=["mse"]) 
                model.compile(loss="mse", optimizer='adam')
                return model
        
        
            feature_dim = x_trn.shape[1]
            input_shape = (feature_dim, )
            model_input = Input(shape=input_shape)
            model = ModelV1(model_input)

            model.summary() 
            model_filepath = '../model/%s-%s-%d-%s-%s-%d-%d-%d.model' % (cancer_type, feature_type, attempt, str(pca), str(dropout), hidden_dims, ae_dim1, ae_dim2 )
            checkpoint = MyCallback(results, model_filepath, data, real_save=True, verbose=1,
                                save_best_only=True,
                                mode='auto', cancer_type = cancer_type , feature_type = feature_type,
                                thr=pca, dropout_prob = dropout, dimension = hidden_dims ,activate='relu',
                                AE1 = ae_dim1, AE2=ae_dim2)
            callbacks_list = [checkpoint]


            history = model.fit(x_trn, s_trn,
                batch_size=batch_size,
                shuffle=True,
                callbacks=callbacks_list,
                epochs=epochs,
                validation_data=(x_dev, s_dev))

            #print("-----History----")
            #print(history.history.keys())
            #print(history.history)
            #print(len(history.history['val_loss']))

            pred_tst = model.predict(x_tst)

            return my_cindex(c_tst, s_tst)(s_tst, pred_tst)
        return inner_SoM

    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump

    bo_dict = { "pca" : (0.98, 0.9999), "dropout" : (0, 0.8), 
        "hidden_dims" : (10, 1000), 
        "ae_dim1" : (100, 1500), 
        "ae_dim2" : (100,700)
    }

    #for k in bo_dict.keys() : 
    #    print(k)
    #    print (bo_dict[k])

    #scoreofModel(**{'ae_dim1': 1138.0196836044008, 'dropout': 0.18242910081095307, 'pca': 0.98912275449631237, 'hidden_dims': 373.61768597111694, 'ae_dim2': 472.20225514485821})

    v1BO = BayesianOptimization(scoreofModel(cancer_type, feature_type, attempt), bo_dict,verbose=True)
    
    v1BO.explore({
        "pca" : [0.98, 0.1, 0.9999],
        "dropout" : [0, 0.2,0.8], 
        "hidden_dims" : [10, 200, 1000], 
        "ae_dim1" : [100, 300, 1500], 
        "ae_dim2" : [100,100,700],
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

    with open('./BO_Result_'+cancer_type+'.txt','at' ) as f:
        
        params =v1BO.res['all']['params']
        values = v1BO.res['all']['values']
        keys = params[0].keys() 

        for i in range(2) : 
            line = [cancer_type, feature_type] 
          
            for k in keys : 
                line.append(str(params[i][k]))
            line.append(str(values[i]))
            f.write('\t'.join(line)+'\n')
    

            


        
           

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default="3", choices=["0", "1"], type=str)
    parser.add_argument('-ct', default="LUAD", choices=['LUAD', 'LUSC'], type=str) # cancer_type
    parser.add_argument('-ft', default="RPM_miRNA.hg19.mirbase20",
                        choices=['normalized_RNA-seq.hg19', 'RPM_miRNA.hg19.mirbase20', 'betaValue_methyl.hg19.sd15'],
                        type=str)  # feature_type
    parser.add_argument('-t', default=0, choices=range(10), type=int)

    args = parser.parse_args()


    for ct in ['LUAD', 'LUSC'] :
        run(args.g,ct, args.ft, args.t)
    
    
"""