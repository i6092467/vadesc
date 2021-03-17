import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import keras
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib





############### PROTOTYPES SAMPLING UTILITY FUNCTIONS #####################################


def Prototypes_sampler(cluster, X, pcz, sample_size, p_threshold):
    
    #X = pd.DataFrame(X)
    # Function to extract prototypes from X assigned to cluster c with high probability (>= pcz_threshold)
    High_p_c_df = pd.DataFrame(pcz.loc[(pcz.iloc[:,cluster] > p_threshold), cluster])
    
    # make sure we sample always the same prototypes for each cluster 
    np.random.seed(seed=42)
    
    # Check if there are enough observations with high probability to sample for the given cluster
    if len(High_p_c_df) <= sample_size:
        id_X = High_p_c_df.index
    else:
        id_X = High_p_c_df.sample(n=sample_size).index
    
    Prototypes_c = X.iloc[id_X]

    return Prototypes_c, id_X



def extract_prototypes_list(X, clusters_labs, pcz, n_prototypes, p_threshold):
    proto_id_list = []
    
    for cluster in clusters_labs:
        df, proto_id = Prototypes_sampler(cluster, X, pcz, sample_size = n_prototypes, p_threshold = p_threshold)
        proto_id_list.append(proto_id)
        
    return proto_id_list




def build_prototypes_ds(X, num_clusters, proto_id_list): 
    
    Prototypes_ds = pd.DataFrame()
    proto_labels = []

    for i in range(0,num_clusters):
        df = X.iloc[proto_id_list[i],:]
        lab = np.full((np.shape(df)[0],), i)
        
        Prototypes_ds = pd.concat([Prototypes_ds, df], axis=0)
        proto_labels = np.append(proto_labels, lab)
        

    return Prototypes_ds, proto_labels




############### HEMO DATA UTILS #################

def import_hemo_covnames():
    cov_names = ['ageStart', 'myspKtV', 'myektv', 'UFR_mLkgh', 'zwtpost',
       'CharlsonScore', 'diabetes', 'cardiovascular', 'ctd', 'mean_albumin',
       'mean_nPCR', 'mean_ldh', 'mean_creatinine', 'mean_hematocrit',
       'mean_iron', 'mean_neutrophils', 'mean_lymphocytes', 'mean_rdw',
       'mean_rbc', 'mean_ag_ratio', 'mean_caxphos_c', 'mean_hemoglobin',
       'mean_pth', 'mean_uf', 'mean_uf_percent', 'mean_idwg_day',
       'mean_preSBP', 'mean_postSBP', 'mean_lowestSBP', 'TBWchild', 'TBWadult',
       'BSA', 'cTargetDryWeightKg', 'WeightPostKg', 'spktv_cheek_BSA',
       'spktv_cheek_W067', 'spktv_cheek_W075', 'spktv_watson_BSA',
       'spktv_watson_W067', 'spktv_watson_W075', 'tidwg2', 'tuf_percent',
       'PatientGender_F', 'PatientRace4_African',
       'PatientRace4_Caucasian', 'PatientRace4_Hispanic',
       'USRDS_class_Cystic/hereditary/congenital diseases',
       'USRDS_class_Diabetes', 'USRDS_class_Glomerulonephritis',
       'USRDS_class_Hypertensive/large vessel disease',
       'USRDS_class_Interstitial nephritis/pyelonephritis',
       'USRDS_class_Miscellaneous conditions ', 'USRDS_class_Neoplasms/tumors',
       'USRDS_class_Secondary glomerulonephritis/vasculitis',
       'fspktv4_(1.39,1.56]', 'fspktv4_(1.56,1.73]', 'fspktv4_(1.73,3.63]',
       'fspktv4_[0.784,1.39]']
    return cov_names


def HemoData_preparation(X):
    
    cov_names = import_hemo_covnames()
    X = pd.DataFrame(X)
    X.columns = cov_names
    
    cov_to_eliminate = ['UFR_mLkgh', 
                        'mean_uf',
                        'mean_idwg_day',
                        'mean_postSBP',
                        'mean_lowestSBP',
                        'TBWchild',
                        'TBWadult',
                       'spktv_watson_W067',
                        'spktv_watson_W075',
                        'spktv_watson_BSA',
                        'spktv_cheek_BSA',
                        'spktv_cheek_W075',
                        'tidwg2',
                        'tuf_percent',
                        'fspktv4_(1.39,1.56]', 
                        'fspktv4_(1.56,1.73]', 
                        'fspktv4_(1.73,3.63]',
                        'fspktv4_[0.784,1.39]']
       
    X = X.drop(cov_to_eliminate, axis=1)
    cov_names = X.columns.values
    
    return X.values, cov_names

    
    
    
    
########## PLOTTING UTILS ############################################   
    


def prepare_summary_plot_data(global_shaps, top_n, prototypes_ds_original, cluster_labels, feature_names): 

    most_rel_shaps_ds = global_shaps.nlargest(top_n)

    # We extract the id of the most relevant features to retrieve the columns from the raw input data. 
    # This passage is needed to plot the original features distribution in the two clusters of prototypes.
    id_most_rel = most_rel_shaps_ds.index
    
    Proto_mostRel_f_ds = prototypes_ds_original.iloc[:,id_most_rel]
    
    Plot_df = pd.concat([Proto_mostRel_f_ds, pd.DataFrame(cluster_labels, columns=["c"])], axis=1)
    top_feature_names = feature_names[id_most_rel]
    shap_bar_values = most_rel_shaps_ds.tolist()
    
    return top_feature_names, shap_bar_values, Plot_df



def plot_topN_features(Plot_df, top_n, top_feature_names, shap_bar_values, unit_measures):
    
    CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

    number_gp = top_n

    def ax_settings(ax, var_name, unit_measure):

        ax.set_yticks([])

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.spines['bottom'].set_edgecolor('#444444')
        ax.spines['bottom'].set_linewidth(2)

        ax.set_xlabel(unit_measure, fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        #ax.set_xticklabels(ax.get_xticklabels(), fontsize=4)

        ax.text(-0.2, 0.1, var_name, fontsize=17, transform = ax.transAxes) 
        return None


    # Manipulate each axes object in the left.
    fig = plt.figure(figsize=(18,21))

    gs = matplotlib.gridspec.GridSpec(nrows=number_gp, 
                           ncols=2, 
                           figure=fig, 
                           width_ratios= [3, 1],
                           height_ratios= [1]*number_gp,
                           wspace=0.05, hspace=0.6
                          )
    ax = [None]*(number_gp)

    # Create a figure, partition the figure into boxes, set up an ax array to store axes objects, and create a list of features.  


    for i in range(number_gp):
        ax[i] = fig.add_subplot(gs[i, 0])

        ax_settings(ax[i], str(top_feature_names[i]), str(unit_measures[i]))    

        sns.histplot(data=Plot_df[(Plot_df['c'] == 0)].iloc[:,i], ax=ax[i], stat = 'density', color=CB_COLOR_CYCLE[1], legend=False, alpha=0.6, linewidth=0.1)
        sns.histplot(data=Plot_df[(Plot_df['c'] == 1)].iloc[:,i], ax=ax[i], stat = 'density', color=CB_COLOR_CYCLE[0], legend=False, alpha=0.6, linewidth=0.1)



        #if i < (number_gp - 1): 
        #    ax[i].set_xticks([])

        if i == (number_gp-1):
            ax[i].text(0.2, -1, 'Covariates Distribution across Clusters', fontsize=18, transform = ax[i].transAxes)


    ax[0].legend(['Cluster 1', 'Cluster 2'], facecolor='w', loc='upper left', fontsize=15)




    for i in range(number_gp):
        ax[i] = fig.add_subplot(gs[i, 1])

        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].barh(0, shap_bar_values[i], color=CB_COLOR_CYCLE[-3], height=0.8, align = 'center')
        ax[i].set_xlim(0 , 0.015)
        ax[i].set_yticks([])
        ax[i].set_ylim(-1,1)  


        if i < (number_gp - 1): 
            ax[i].set_xticks([])
            ax[i].spines['bottom'].set_visible(False)

        if i == (number_gp-1): 
            ax[i].spines['bottom'].set_visible(True)
            ax[i].tick_params(axis='x', labelrotation= 45, labelsize=13)      
            ax[i].text(-0.01, -1, 'Mean(|Shapley Value|)', fontsize=18, transform = ax[i].transAxes) 



    return fig



    
    
    
 
