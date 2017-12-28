from keras.layers import Dense, Dropout, Activation, Input, Lambda, BatchNormalization
from keras.optimizers import SGD, Nadam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from keras.models import Model, load_model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import pickle
import variables
import numpy as np
import os 
import pandas as pd 
import itertools
from keras.layers.merge import Add, Concatenate

model_path = "./model/RN/"

def ae_to( cancer_type, feature_type, x):

    

    encoding_dim = variables.rn_dim
    ae_path = model_path +"AE_%s_%s_%d.h5py" %(cancer_type, feature_type, encoding_dim)
    e_path = model_path +"E_%s_%s_%d.h5py" %(cancer_type, feature_type, encoding_dim)
    if os.path.exists(e_path):
        encoder = load_model(e_path)
        return encoder.predict(x)
    
    input_layer = Input(shape=(x.shape[1],))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(x.shape[1], activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x, x, epochs=100, batch_size = 32, 
    shuffle=True,validation_split=0.2 )

    reduced_x = encoder.predict(x).astype(np.float64)
    autoencoder.save(ae_path, overwrite=True)
    encoder.save(e_path, overwrite=True)

    return reduced_x

"""
    load data
"""
def load_data_by_sample(cancer_type):
    # load pickle file 
    cancer_type_list = ['LUAD', 'LUSC']
    feature_type_list =  ['betaValue_methyl.hg19.sd15', 'normalized_RNA-seq.hg19', 'RPM_miRNA.hg19.mirbase20'] #, 'betaValue_methyl.hg19']
    
    def make_query(f1,f2,len):
        q = [0,0,0]
        q[feature_type_list.index(f1)] = 1 
        q[feature_type_list.index(f2)] = 1
        qs = []
        for i in range(len):
            qs.append(q)
        return qs 


    df_dict = {}  
    label_dict = {} 
    # load already concated pickle file (by pid)
    sd = 17
    if cancer_type in cancer_type_list: # TODO : remove it 
        for feature_type in feature_type_list: 
            with open('../test_bong/data/%s.%s.pkl' % ( cancer_type, feature_type), 'rb') as handle:
                labels = pickle.load(handle)
                label_dict[cancer_type] = labels 
                x = pickle.load(handle)
                y = pickle.load(handle)
                #print "x.shape ", x.shape
                #print "y.shape", y.shape

                x_reduced = ae_to(cancer_type, feature_type, x)
                df_new = pd.DataFrame(data = x_reduced)
                df_new.columns = range(variables.rn_dim)
                df_new['c'] = [t[0] for t in y]
                #print df_new['c'].values
                df_new['y'] = [t[1] for t in y]
                df_new.index = labels 
                print "df_new.shape ", df_new.shape
                #print df_new.head()
                df_dict[cancer_type+"_"+feature_type] = df_new
    del(df_new)
    
    rn_dataframes = []
    df_all = pd.DataFrame() 
    l = [feature_type_list, feature_type_list] 
    if cancer_type in cancer_type_list: 
        for f1,f2 in list(itertools.product(*l)):
            
            if f1 != f2 : 
                print f1, f2
                df_temp = pd.merge(df_dict.get(cancer_type+"_"+f1).iloc[:,:variables.rn_dim],
                    df_dict.get(cancer_type+"_"+f2), right_index=True, left_index=True)
                df_temp['index'] = df_temp.index

                df_temp =df_temp.drop_duplicates('index', 'first')
                df_temp = df_temp.iloc[:,:-1]
                print "df_temp.shape ", df_temp.shape 
                if variables.rn_query : 
                    queries = pd.DataFrame(data=make_query(f1,f2,df_temp.shape[0]))
                    queries.columns = feature_type_list
                    queries.index = df_temp.index #label_dict.get(cancer_type)
                    df_temp = pd.concat([df_temp, queries], axis=1) 
                    print "df_tmp.shape aft queries ", df_temp.shape

                df_all = pd.concat([df_all,df_temp])

                print "df_all.shape ", df_all.shape
        df_all.to_csv('../test_bong/data/rn_%d.txt' % variables.rn_dim,sep='\t')
        #rn_dataframes.append(df_all)
        #df_all = pd.DataFrame()
    #return rn_dataframes 
    return df_all

    
    
                
    # according to the pid, pair data and put relation label 


"""
    Build Model
"""

def construct_model_RNs(n_input_width, rn_args, n_samples):
    
    MLP_unit = 256
    DENSE_unit_1 = 512
    DENSE_unit_2 = 1024 

    RN_FIX_F_NUM = 100
    RN_REL_F_NUM = n_samples ### @@ 
    RN_REL_F_SIZE = 100 

    
    #n_feature = rn_args["n_feature"]
    g_MLP_layers = rn_args["g_MLP_layers"]
    f_MLP_layers = rn_args["f_MLP_layers"]

    
    # TODO : Lambdas? slice 
    def stack_layer(layers):
        def f(x):
            print(x)
            for k in range(len(layers)):
                x = layers[k](x)
            return x
        return f

    def get_MLP(MLP_units):
        print "mlp units ", MLP_units
        r = []
        for MLP_unit in MLP_units:
            s = stack_layer([
                Dense(MLP_unit),
                BatchNormalization(),
                Activation('relu'),
            ])
            r.append(s)
        return stack_layer(r)


    def slice_fix(t):
        if variables.rn_relation_by == "FEATURE":
            return t[:,0:RN_FIX_F_NUM]
        else : 
            return t[:,:] ##@@

    def bn_dense(x, unit, layer_name):
        y = Dense(unit, name = layer_name)(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Dropout(0.5)(y)
        return y

    inputs = Input((n_input_width, ))
   
    relations = [] 
    concat = Concatenate() 
    con_cnt = 0 

    if variables.rn_relation_by == "SAMPLE":
        # relation pair ex (m rna, mirna), (m rna, dna meth)
       
        def slice_first(t):
            #return t.iloc[:,0:variables.rn_dim]
            return tf.slice(t, [0,0], [tf.shape(t)[0],variables.rn_dim])
        def slice_second(t):
            #return t.iloc[:,variables.rn_dim:]
            return tf.slice(t,[0,variables.rn_dim], [tf.shape(t)[0],variables.rn_dim*2])
            #TODO : query 
        def q(t): 
            return tf.slice(t, [0,variables.rn_dim*2], [tf.shape(t)[0], tf.shape(t)[1]])
        
        # @@ 
        base_layer = Lambda(slice_first)
        others = []
        others.append(Lambda(slice_second))
        if variables.rn_query:
            others.append(Lambda(q))
        base_feature = base_layer(inputs)
        other_features = [] 
        for i in others :
            other_features.append(i(inputs))

        relations = []
        for other_feature in other_features : 
            relations.append(Concatenate()([base_feature, other_feature]))
        #if variables.rn_query == True :
            #pairts.append embedding 
        
    
    
    g_MLP = get_MLP(g_MLP_layers)

    mid_relations = [] 
    for r in relations :
        mid_relations.append(g_MLP(r))
    combined_relation = Add()(mid_relations)
    rn = bn_dense(combined_relation, DENSE_unit_1, 'dense_feature1')
    out = Dense(1)(rn)

    model = Model(inputs = [inputs], outputs=out)
    return model 



def train_RNs(X_train, Y_train, rn_args) : 

    # X_train should already be pairs

    def loss_func(y_true, y_pred):
        return ktf.sqrt(ktf.mean(ktf.square(y_pred-y_true)))

    learning_rate = rn_args["learning_rate"]
    epochs = rn_args["epochs"]
    batch_size = rn_args["batch_size"]
    model_name = rn_args["model_name"]

    # Build model
    n_input_width = len(X_train[0])


    model = construct_model_RNs(n_input_width, rn_args, 0)
    optimizer = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=loss_func, optimizer=optimizer)


     # Callbacks
    best_model_path= model_path + "%s_best_weight.hdf5" % (model_name)
    save_best_model = ModelCheckpoint(best_model_path, monitor="val_loss", verbose=0, save_best_only=True, mode='min')
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.001, cooldown=0, min_lr=0)

    # Train
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, callbacks=[save_best_model, reduce_lr_on_plateau])

    # Load best weight
    model.load_weights(best_model_path)
    return model



