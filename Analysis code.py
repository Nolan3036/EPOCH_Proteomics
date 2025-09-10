#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# # function

# ## Univariate analysis, volcano plot, and gene enrichment analysis

# In[3]:


import math
def FC(d1, d2):
    #calculate log2 fold change
    #d2 is baseline
    return np.mean(np.log2(d1))-np.mean(np.log2(d2))
def map_color(df):
    if df["-log10(fdr_pv)"]<-np.log10(0.05) or (df["log2(FC)"]==0):
        return "ns"
    elif df["log2(FC)"]<0:
        return "Down"
    elif df["log2(FC)"]>0:
        return "Up"


# In[4]:


from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
from operator import itemgetter 
import gseapy as gp
from gseapy import dotplot,barplot
from adjustText import adjust_text
import matplotlib as mpl
def volcano_plot_enrichment_analysis(proteomics1,proteomics2,ftable,size=5,filename=""):
    """
    proteomics1:
    proteomics2:control group (baseline group)
    ftable:protein and gene information
    """
    #set up univariate p value and fold change vector
    result = np.full(proteomics1.shape[1],np.nan)
    fc=np.full(proteomics1.shape[1],np.nan)
    #calculate univariate p value and fold change
    for i,col in enumerate(proteomics1.columns):
        stat,pv=mannwhitneyu(proteomics1[col].values,proteomics2[col].values)
        result[i]=pv
        fc[i]=FC(proteomics1[col].values,proteomics2[col].values)
    #multiple testing correction
    reject,pvals_corrected,alphacSidak,alphacBonf = multipletests(result, alpha=0.05, method='fdr_bh')
    #build result dataframe
    result_df=pd.DataFrame({"AptName":proteomics1.columns,
                            "-log10(fdr_pv)":-np.log10(pvals_corrected),
                            "log2(FC)":fc})
    result_df=result_df.merge(ftable[["AptName","EntrezGeneSymbol"]], on='AptName')
    result_df['color'] = result_df.apply(map_color, axis = 1)

#     temp_df=result_df[result_df['color']!="ns"].sort_values("-log10(fdr_pv)",ascending=False)
    
    ##volcano plot
    plt.figure(figsize = (4,4))
    sns.scatterplot(data = result_df, x = "log2(FC)", y = '-log10(fdr_pv)',
                    hue = 'color', hue_order = ["Up", 'ns',"Down"],
                    palette = ['red','lightgrey',"blue",])
    sns.despine(top=True, right=True)
    plt.axhline(-np.log10(0.05), zorder = 0, c = 'k', ls = '--')
    plt.axvline(0, zorder = 0, c = 'k', ls = '--')
    plt.legend(title="",bbox_to_anchor=(1.04, 0.5), loc="center left",frameon=False)
    xabs_max = abs(max(plt.xlim(), key=abs))
    plt.xlim(-xabs_max,xabs_max)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("log2(FC)",fontsize = 14)
    plt.ylabel('-log10(fdr_pv)',fontsize = 14)
    plt.show()
    plt.close()
    ##enrichment analysis
    #get background gene list(remove duplicates, missing value)
    glist_backgroud=ftable[(~ftable.duplicated(subset=['EntrezGeneSymbol']))&(~ftable.EntrezGeneSymbol.isnull())].EntrezGeneSymbol.tolist()
    #over representation
    enr = gp.enrichr(gene_list=result_df[(result_df["color"]!="ns")&
                                         (~result_df.EntrezGeneSymbol.isnull())].EntrezGeneSymbol.tolist(), 
                 gene_sets=['WikiPathways_2019_Human'],# Here you can choose whatever database you like
                 #gene_sets=['KEGG_2021_Human']
                 background=glist_backgroud,
                 organism='human')
    dotplot(enr.results,
              column="Adjusted P-value",
#               x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              top_term=5,
            size=size,
              title = "WikiPathways_2019_Human",
              # xticklabels_rot=45, # rotate xtick labels
              show_ring=True, # set to False to revmove outer ring
              marker='o')
    result_df.to_csv(f'{filename}_univariate_test.csv',index=False)


# ## ML

# In[5]:


import shap
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import KFold


# In[6]:


def ML(model_func,df,y,params={},CV_round=10,k=10):
    """ Do Kfold cross validation with the specified arguments
    model_func: function. Constructor of the model.
    df: proteomics
    y: numpy.array
    """
    #set up test result array
    test_result = np.full((df.shape[0],CV_round),np.nan)
    # Built in K-fold function in Sci-Kit Learn
    np.random.seed(88)
    KFold_rs=np.random.randint(low=0,high=(2**32 - 1),size=CV_round)
    np.random.seed(888)
    model_rs=np.random.randint(low=0,high=(2**32 - 1),size=(CV_round,k))
    for rounds in tqdm(range(CV_round),desc = 'Round'):
        kf = KFold(n_splits=k,shuffle=True,random_state=KFold_rs[rounds])
        for (fold, (train_index, test_index)) in tqdm(enumerate(kf.split(df.values)),desc = 'fold'):
            train_X, test_X = df.iloc[train_index], df.iloc[test_index]
            train_y, test_y = y[train_index], y[test_index]
            train_X=train_X.astype(float)
            test_X=test_X.astype(float)
            #model,trainer set up | train, predict
            model=model_func(**params,random_state=model_rs[rounds,fold]).fit(train_X,train_y)
            test_prediction=model.predict_proba(test_X)[:,1]
            for j, sample in enumerate(test_index):
                test_result[sample,rounds]=test_prediction[j]
    #average result
    test_result=np.mean(test_result,axis=1)
    return test_result


# ## Covariate adjustment

# In[32]:


import statsmodels.api as sm
from scipy.stats import pointbiserialr,spearmanr
import pingouin as pg
def covariate_adjustment(binary_outcome,model_probability,covariate_vector=None,rounds=10000):
    """
    binary_outcome: true binary label. numpy array.
    model_probability: probability generated by model. numpy array.
    covariate_vector: when None, generate statistics without adjustment.
    rounds: bootstrap round number 
    return correlation coefficient, p value, and 95% confidence interval
    """
    #without adjustment
    if covariate_vector is None:
        # point biserial correlation
        r,p=pointbiserialr(binary_outcome,model_probability)
        # bootstrap for confidence interval
        r_bootstrap=np.random.random(rounds)
        for i in range(rounds):
            indices = np.random.choice(binary_outcome.shape[0], binary_outcome.shape[0], replace=True)
            x_bootstrap = model_probability[indices]
            y_bootstrap = binary_outcome[indices]
            r_bootstrap[i]=pointbiserialr(y_bootstrap,x_bootstrap)[0]
    else:
        model_x = sm.OLS(model_probability, sm.add_constant(covariate_vector))
        model_y = sm.OLS(binary_outcome, sm.add_constant(covariate_vector))
        results_x = model_x.fit()
        results_y = model_y.fit()
        # Calculate residuals
        residuals_x = results_x.resid
        residuals_y = results_y.resid
        # correlation on residual
        r,p=spearmanr(residuals_y,residuals_x)
        # bootstrap for confidence interval
        r_bootstrap=np.random.random(rounds)
        for i in range(rounds):
            indices = np.random.choice(residuals_x.shape[0], residuals_x.shape[0], replace=True)
            x_bootstrap = residuals_x[indices]
            y_bootstrap = residuals_y[indices]
            r_bootstrap[i]=spearmanr(x_bootstrap,y_bootstrap)[0]
    ci=(np.quantile(r_bootstrap, 0.025),np.quantile(r_bootstrap, 0.975))
    return r,p,ci


# # example

# In[17]:


#read data
#proteomics
data=pd.read_csv("proteomics_publication.csv")
#gene information
ftable=pd.read_csv("featuretable.csv")


# In[19]:


#univariate analysis (AP)
volcano_plot_enrichment_analysis(data[(data["Time Point"]=="AP")&(data["Case/Control"]=="Case")].drop(["Time Point","Case/Control",'BMI @ Sampling','GDM','GA or Time since last Delivery'],axis=1),
                                 data[(data["Time Point"]=="AP")&(data["Case/Control"]=="Control")].drop(["Time Point","Case/Control",'BMI @ Sampling','GDM','GA or Time since last Delivery'],axis=1),
                                 ftable,filename="AP_case_control")


# In[20]:


#Classification cases v.s. control (AP)
AP_result=ML(XGBClassifier,
             data[data["Time Point"]=="AP"].reset_index(drop=True).drop(["Time Point","Case/Control",'BMI @ Sampling','GDM','GA or Time since last Delivery'],axis=1),
             data[data["Time Point"]=="AP"]["Case/Control"].replace({"Case":1,"Control":0}).values,
             params={},
             CV_round=10,k=10)


# In[23]:


from sklearn.metrics import roc_auc_score,roc_curve
#calculate AUC
print("AUC: ",roc_auc_score(data[data["Time Point"]=="AP"]["Case/Control"].replace({"Case":1,"Control":0}).values, AP_result))


# In[34]:


#covariate adjustment
r,p,ci=covariate_adjustment(data[data["Time Point"]=="AP"]["Case/Control"].replace({"Case":1,"Control":0}).values,
                     AP_result,
                     data[data["Time Point"]=="AP"]["GDM"].values
                    )


# In[ ]:




