
import sys, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cospar
import clinc.clinc_util as clinc


# TRACERSEQ ANALYSIS

def load_tracerseq_barcode_counts(adata, key, path):

  # load TracerSeq counts file into a pandas dataframe
  df = pd.read_csv(path, dtype='str', delimiter=',', header=None)
  df.columns =['unique_cell_id', 'UniqueTracerID', 'TracerBarcode', 'UMI_counts']
  df['unique_cell_id'] = df['unique_cell_id'].str.replace("A_", "A-").str.replace("T_", "T-").str.replace("G_", "G-").str.replace("C_", "C-")

  # filter the dataframe to only include cell barcodes present in adata
  cells_flag = np.in1d(df['unique_cell_id'], adata.obs['unique_cell_id'].tolist()) 
  df = df.drop(df[~cells_flag].index).reset_index(drop=True)
  df['UniqueTracerID'] = np.unique(df['UniqueTracerID'], return_inverse=True)[1] # 'reset' the UniqueTracerID column

  # create an empty counts matrix 'm': one row for each cell barcode, one column for each unique TracerSeq barcode, entries will be UMI counts
  nCells_adata = len(adata.obs.unique_cell_id)
  nUniqueTracerBarcodes = len(np.unique(df['UniqueTracerID']))
  m = np.zeros((nCells_adata,nUniqueTracerBarcodes))

  # create an empty array 'bcd': one entry for each unique TracerSeq barcode sequence
  bcd = np.array([None] * nUniqueTracerBarcodes)
  
  # populate the m matrix with UMI counts for each cell-TracerSeq barcode pair
  # populate the bcd list with the original TracerSeq barcode sequences
  for r in range(len(df)): 
      this_row = np.where(np.in1d(adata.obs['unique_cell_id'],df['unique_cell_id'][r]))[0][0]
      this_column = int(df['UniqueTracerID'][r]) - 1 # convert to zero-based index
      m[this_row,this_column] = df['UMI_counts'][r]
      bcd[this_column]=(df['TracerBarcode'][r])

  # filter to only include TracerSeq barcodes that comprise a clone (2 cells or more)
  nTracerBarcodes = m.shape[1]
  clones_flag = np.count_nonzero(m, axis=0)>1
  nClones = np.count_nonzero(clones_flag)
  while nTracerBarcodes > nClones:
    m = m[:,clones_flag]
    bcd = bcd[clones_flag]
    nTracerBarcodes = m.shape[1]
    clones_flag = np.count_nonzero(m, axis=0)>1
    nClones = np.count_nonzero(clones_flag)

  print(key, 'nTracerBarcodes:', m.shape[1])
  print(key, 'nTracerCells:', np.count_nonzero(~np.all(m == 0, axis=1)))

  # export to 'TracerSeq' adata.obsm dataframe
  df_export = pd.DataFrame(data = m, index = adata.obs.index.copy(), columns = [key + "_" + bcd])
  if 'TracerSeq' in list(adata.obsm.keys()): # if 'TracerSeq' obsm already exists, append to it
    adata.obsm['TracerSeq'] = pd.concat([adata.obsm['TracerSeq'], df_export], axis = 1)
  else:
    adata.obsm['TracerSeq'] = df_export
  
  # drop duplicate columns, if present
  adata.obsm['TracerSeq'] = adata.obsm['TracerSeq'].T.drop_duplicates().T

  return adata


def plot_cells_vs_barcodes_heatmap(adata, cell_labels_key=None, umi_thresh=0):
  
  X = adata.obsm['TracerSeq']

  # convert TracerSeq counts matrix to boolean based on UMI threshold
  X = (X > umi_thresh)*1
  
  # filter cells with both transcriptome and TracerSeq information
  flag = X.sum(axis = 1) > 0
  X = X[flag]

  # plot a clustered heatmap of cells x barcodes 
  sys.setrecursionlimit(100000) 
  
  # set up cell labels
  if cell_labels_key is not None:
    cell_labels = adata.obs[cell_labels_key]
    cell_label_colors = adata.uns[cell_labels_key + '_colors']
    lut=dict(zip(np.unique(cell_labels),cell_label_colors))
    row_colors = cell_labels.map(lut)
    row_colors = row_colors[flag]
  else:
    row_colors=[]
  
  # generate cluster map with or without cell labels
  if cell_labels_key is not None:
    cg = sns.clustermap(X, 
                        metric='jaccard', cmap='Greys', 
                        cbar_pos=None, 
                        xticklabels=False, yticklabels=False,
                        dendrogram_ratio=0.08, figsize=(6, 8),
                        row_colors=row_colors,
                        colors_ratio=0.02)
  else:
    cg = sns.clustermap(X, 
                    metric='jaccard', cmap='Greys', 
                    cbar_pos=None, 
                    xticklabels=False, yticklabels=False,
                    dendrogram_ratio=0.08, figsize=(6, 8))
  
  # format plot
  cg.ax_heatmap.set_xlabel('Clones')
  cg.ax_heatmap.set_ylabel('Cells')
  for _, spine in cg.ax_heatmap.spines.items():
    spine.set_visible(True) # draws a simple frame around the heatmap
  cg.ax_col_dendrogram.set_visible(False) # hide the column dendrogram 


def plot_state_couplings_heatmap(X, state_IDs=None, title=None, tick_fontsize=10, figsize=8, do_clustering=False, metric='correlation', linkage='average'):   
    
    # Plot a seaborn clustermap of state-state barcode couplings

    if state_IDs is not None:
      X = pd.DataFrame(X, index=state_IDs, columns=state_IDs)
    
    vmax = (np.percentile(X-np.diag(np.diag(X)),95) + np.percentile(X-np.diag(np.diag(X)),98))/2
    vmax = (np.percentile(X-np.diag(np.diag(X)),95) + np.percentile(X-np.diag(np.diag(X)),98))/2
    
    cg = sns.clustermap(X, metric=metric, method=linkage, cmap='viridis', 
                        cbar_pos=None, dendrogram_ratio=0.2, figsize=(figsize,figsize),
                        col_cluster = do_clustering, row_cluster = do_clustering,
                        xticklabels = 1, yticklabels = 1, colors_ratio=0.02, vmax=vmax)  
    
    cg.ax_col_dendrogram.set_visible(False) # hide the column dendrogram
    cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize = tick_fontsize)
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize = tick_fontsize)
    
    plt.title(title)


def get_observed_barcode_couplings(adata, cell_state_key, umi_thresh=1, thresh_min_cells_per_hit=1):
  
  # Calculate 'OBSERVED' barcode couplings between states

  # For all state pairs, sum the number of times a cell with a given TracerSeq barcode was identified in both state j and state k

  # import data
  adata = adata[~adata.obs[cell_state_key].isin(['NaN']),:]
  cell_states = adata.obs[cell_state_key]
  X = adata.obsm['TracerSeq']

  # convert TracerSeq counts matrix to boolean based on UMI threshold
  X = np.array(X >= umi_thresh)*1
  
  # filter to cells with both state (transcriptome) and TracerSeq information, filter out states with zero hits
  flag = X.sum(axis = 1) > 0
  X = X[flag]
  cell_states = cell_states[flag]
  coupled_state_IDs = np.unique(cell_states)
  nStates = len(coupled_state_IDs)
  
  # compute the observed couplings matrix
  X_obs = np.zeros((nStates,nStates))  
  for j in range(nStates):    
    cells_in_state_j = np.array(coupled_state_IDs[j] == cell_states) # index the cells assigned to this particular j state
    clone_hits_in_state_j = sum(X[cells_in_state_j,:]) >= thresh_min_cells_per_hit
    for k in range(j,nStates): # calculate upper triangle only to save time
      cells_in_state_k = np.array(coupled_state_IDs[k] == cell_states) # index the cells assigned to this particular k state
      clone_hits_in_state_k = sum(X[cells_in_state_k,:]) >= thresh_min_cells_per_hit    
      X_obs[j,k] = sum(clone_hits_in_state_j & clone_hits_in_state_k)
  X_obs = np.maximum(X_obs,X_obs.transpose()) # re-symmetrize the matrix

  return X_obs, coupled_state_IDs


def get_oe_barcode_couplings(X_obs):

  # Calculate 'OBSERVED/EXPECTED' barcode couplings between states

  # Given the observed barcode couplings matrix, coupling frequencies expected by random chance are the outer product of the column sums and row sums normalized by the total.

  X = np.array(X_obs)
  X_expect = X.sum(0, keepdims=True) * X.sum(1, keepdims=True) / X.sum()
  X_oe = X_obs/X_expect

  return X_oe


def plot_clinc_neighbor_joining(output_directory, node_groups, celltype_names, X_history, merged_pairs_history, node_names_history):
    
    # updated version of clinc function

    fig,axs = plt.subplots(1,len(X_history))
    
    for i,X in enumerate(X_history):
        #vmaxx = 40
        #axs[i].imshow(X,vmax=vmaxx)
        axs[i].imshow(X)
        axs[i].grid(None)
        ii,jj = merged_pairs_history[i]
        axs[i].scatter([jj],[ii],s=100, marker='*', c='white')

        column_groups = [node_groups[n] for n in node_names_history[i]]
        column_labels = [' + '.join([celltype_names[n] for n in grp]) for grp in column_groups]
        axs[i].set_xticks(np.arange(X.shape[1])+.4)
        axs[i].set_xticklabels(column_labels, rotation=90, ha='right')
        axs[i].set_xlim([-.5,X.shape[1]-.5])
        axs[i].set_ylim([X.shape[1]-.5,-.5])
    
    fig.set_size_inches((100,100))
    #plt.savefig(output_directory+'/neighbor_joint_heatmaps.pdf')


