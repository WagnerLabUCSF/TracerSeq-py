
import sys, random, warnings, yaml
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import networkx as nx

import Levenshtein
import cospar
import clinc.clinc_util as clinc


# TRACERSEQ PREPROCESSING

def process_tracerseq_csv_to_counts(path_to_run_params_yaml):

    # load run parameters from yaml config file
    with open(path_to_config_yaml) as f:
      run_params = yaml.load(f, Loader=yaml.FullLoader)

    n_files = len(run_params['Files Dictionary'])
    print('Processing TracerSeq counts from', str(n_files), 'libraries')

    # Configure some global settings
    matplotlib.rcParams['figure.dpi'] = 200
    matplotlib.rcParams['figure.figsize'] = [4,4]
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Step 1: Import the entire set of TracerSeq CSV file(s) into a dataframe, appending library prefices to CB columns
    df1 = pd.DataFrame()
    for prefix, file in run_params['Files Dictionary'].items():
      tmp_df = pd.read_csv(run_params['Input File Path'] + file, compression='gzip')
      tmp_df['CB_corrected'] = prefix + '_' + tmp_df['CB_corrected'].astype(str)
      df1 = df1.append(tmp_df, ignore_index=True); del tmp_df
    run_log = {'# TracerSeq Reads Loaded': int(len(df1))}
    print('Done loading barcode csv files (step 1/6)')

    # Step 2: Identify and filter out reads that lack flanking sequences or barcode components
    flag_empty_tracerseq_barcode = df1['TracerSeq'].str.contains(run_params['Barcode Flank Left']+run_params['Barcode Flank Right'])
    flag_no_left = ~df1['TracerSeq'].str.contains(run_params['Barcode Flank Left'])
    flag_no_right = ~df1['TracerSeq'].str.contains(run_params['Barcode Flank Right'])
    flag_no_CB = df1['CB_corrected'].str.contains('-')
    flag_no_UMI = df1['UMI_raw'].str.contains('-')
    flag_filter = flag_empty_tracerseq_barcode | flag_no_left | flag_no_right | flag_no_CB | flag_no_UMI
    df1 = df1[~flag_filter].reset_index().copy()
    print('Done filtering reads (step 2/6)')

    # Log some stats
    run_log['# TracerSeq Reads After Filtering'] = int(len(df1))
    run_log['Fraction Reads Remaining After Filtering'] = round(run_log['# TracerSeq Reads After Filtering']/run_log['# TracerSeq Reads Loaded'],2)
    run_log['Fraction Reads Lacking Tracer Barcode'] = round(np.sum(flag_empty_tracerseq_barcode)/run_log['# TracerSeq Reads Loaded'],2)
    run_log['Fraction Reads Lacking Cell Barcode'] = round(np.sum(flag_no_CB)/run_log['# TracerSeq Reads Loaded'],2)
    run_log['Fraction Reads Lacking UMI'] = round(np.sum(flag_no_UMI)/run_log['# TracerSeq Reads Loaded'],2)
    run_log['Fraction Reads Missing Left Flank'] = round(np.sum(flag_no_left)/run_log['# TracerSeq Reads Loaded'],2)
    run_log['Fraction Reads Missing Right Flank'] = round(np.sum(flag_no_right)/run_log['# TracerSeq Reads Loaded'],2)

    # Step 3: Trim and filter TracerSeq barcodes

    # Step 3a: Trim the TracerSeq barcode read to remove flanking sequences 
    df1['TracerSeq'] = df1['TracerSeq'].str.split(pat=run_params['Barcode Flank Left'], expand=True)[1]
    df1['TracerSeq'] = df1['TracerSeq'].str.split(pat=run_params['Barcode Flank Right'], expand=True)[0]
    df1.rename(columns={'TracerSeq': 'TracerSeq_trimmed'}, inplace=True)
    df1['TracerSeq_barcode_length'] = df1['TracerSeq_trimmed'].str.len()

    # Plot a histogram of TracerSeq barcode lengths, which should be centered at 20bp
    df1['TracerSeq_barcode_length'] = df1['TracerSeq_trimmed'].str.len()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(df1['TracerSeq_barcode_length'], bins=range(0,40), log=True)
    ax.set_title('TracerSeq Barcode Lengths')
    ax.set_xlabel('Barcode Length (bp)')
    ax.set_ylabel('# Trimmed Barcode Reads (log)')
    plt.axvline(x = 15, color = 'red', label = 'thresh', linestyle='-', linewidth=1)
    plt.axvline(x = 25, color = 'red', label = 'thresh', linestyle='-', linewidth=1)
    fig.tight_layout()
    plt.show()
    plt.close()
    
    # Step 3b: Filter out trimmed TBs that are too long or too short
    df1 = df1[df1['TracerSeq_barcode_length'] > 15].copy() # Filter out any severely truncated TBs (rare)
    df1 = df1[df1['TracerSeq_barcode_length'] < 25].copy() # Filter out any severely elongated TBs (rare)
    print('Done trimming TB barcodes (step 3/6)')

    # Plot and filter read counts for all unique TB:CB:UMI barcode sets
    df2 = df1.value_counts(subset=['TracerSeq_trimmed', 'CB_corrected', 'UMI_raw'], sort=True).to_frame('read_counts').reset_index().copy()
    del df1
    counts = df2['read_counts']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(counts, bins=np.logspace(0, 7, 100), weights=counts / sum(counts))
    ax.set_xscale('log')
    ax.set_title('TB:CB:UMI Set Abundance')
    ax.set_xlabel('Read Counts')
    ax.set_ylabel('Fraction of Total Reads')
    plt.axvline(x = run_params['Read Count Thresh'], color = 'red', label = 'thresh', linestyle='-', linewidth=1)
    fig.tight_layout()
    plt.show()
    plt.close()
    df2 = df2[df2['read_counts']>=run_params['Read Count Thresh']].copy() # perform filtering

    # Step 4: Perform TracerSeq barcode correction

    # Step 4a: Determine read counts for each unique TB
    barcodes_use, barcode_ind, barcode_counts_use = np.unique(df2['TracerSeq_trimmed'], return_counts=True, return_inverse=True)
    run_log['# Unique TB:CB:UMI Barcode Sets Detected'] = len(df2['TracerSeq_trimmed'])
    run_log['# Unique TBs Detected'] = len(barcodes_use)
    # Use a dataframe to organize TB correction results
    correction_df = pd.DataFrame({'Barcode': barcodes_use, 'Counts': barcode_counts_use})
    # Get edit distance and adjacency matrices between all pairs of TBs
    edit_dist_all = np.zeros(shape=(len(correction_df), len(correction_df)))
    adj_mat = np.zeros(shape=(len(correction_df), len(correction_df)))
    for j, bcj in enumerate(correction_df['Barcode']):
      for k, bck in enumerate(correction_df['Barcode']):
        edit_dist_all[j,k] = Levenshtein.distance(bcj, bck)
        if edit_dist_all[j,k] <= run_params['Edit Distance Threshold']:
          # No abundance criteria (any clade connected via edges with edit distance 1 is valid)
          adj_mat[k,j] = 1
          # Apply differential abundance criteria (UMI-tools convention of 2k-1)
          #if correction_df['Counts'][j] >= (correction_df['Counts'][k]*2)-1:
            #adj_mat[k,j] = 1

    # Step 4b: Make a graph
    G = nx.Graph()
    np.fill_diagonal(adj_mat, 0)
    G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)

    # Step 4c: Find the graph connected components / clades
    clade_ind = [list(cc) for cc in sorted(nx.weakly_connected_components(G), key=len, reverse=True)]
    run_log['# TB Clades Detected'] = int(nx.number_weakly_connected_components(G))

    # Step 4d: Update dataframe w/ clade assignments
    correction_df['Clade'] = pd.Series(dtype='str')
    for n, node_list in enumerate(clade_ind):
      correction_df.loc[node_list, 'Clade'] = n

    # Step 4e: Build a barcode correction dictionary
    correction_df['Barcode_Corrected'] = pd.Series(dtype='str')
    edit_dist_clades = [] # Keep track of the distances between barcodes in the same clade
    for c in np.unique(correction_df['Clade']): # Get barcodes from each clade, sort by abundance
      flag = correction_df['Clade']==c
      dt_tmp = correction_df[flag].reset_index().copy()
      dt_tmp = dt_tmp.sort_values('Counts', ascending=False)
      parent_barcode_seq = dt_tmp['Barcode'].iloc[0] # parent reference is the most abundant barcode in each clade
      edit_dist_correction = [] # Keep track of the distance from each barcode to the clade parent
      for j, bcj in enumerate(dt_tmp['Barcode']):
        edit_dist_correction.append(Levenshtein.distance(parent_barcode_seq, dt_tmp['Barcode'][j]))
        for k, bck in enumerate(dt_tmp['Barcode']):
          edit_dist_clades.append(Levenshtein.distance(bcj, bck))
      # Copy results for this clade back to the main dataframe
      correction_df.loc[flag, 'Correction_Distance'] = edit_dist_correction
      correction_df.loc[flag, 'Barcode_Corrected'] = parent_barcode_seq

    # Step 4f: Use a dictionary to perform barcode correction
    correction_dict = pd.Series(correction_df.Barcode_Corrected.values,index=correction_df.Barcode).to_dict()
    df2['TracerSeq_corrected'] = [correction_dict[key] for key in df2['TracerSeq_trimmed']]

    # Plot edit distance histograms
    bins = np.linspace(0, 21, num=22)
    h1,bins = np.histogram(edit_dist_all, bins=bins)
    h2,_ = np.histogram(correction_df['Correction_Distance'], bins=bins)
    h3,_ = np.histogram(edit_dist_clades, bins=bins)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    p1 = sns.barplot(x=bin_centers, y=np.log10(1+h1), color='lightgrey', alpha=1, width=0.8, label='all')
    p1 = sns.barplot(x=bin_centers, y=np.log10(1+h3), color='tab:blue', alpha=1, width=0.8, label='within clade')
    p1 = sns.barplot(x=bin_centers, y=np.log10(1+h2), color='tab:orange', alpha=1, width=0.8, label='to parent')
    p1.set_xticks(np.int0(np.linspace(0, 20, num=11)), labels=np.int0(np.linspace(0, 20, num=11)))
    p1.set_xlabel('Edit Distance')
    p1.set_ylabel('# Barcode Pairs (log10)')
    plt.legend(loc='upper right')
    #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()

    # Plot correction graph
    plt.figure(1, figsize=(5, 5))
    pos = nx.nx_agraph.graphviz_layout(G, prog='neato') #'fdp' 'neato' 'sfdp'
    C = (G.subgraph(c) for c in nx.weakly_connected_components(G))
    subgraph_sizes_all = correction_df['Counts']
    subgraph_sizes_all = 50*(subgraph_sizes_all - np.min(subgraph_sizes_all)) / (np.max(subgraph_sizes_all) - np.min(subgraph_sizes_all)) + 5
    for g in C:
        subgraph_sizes = np.array(subgraph_sizes_all[list(g.nodes)])
        subgraph_colors = np.array(np.floor(subgraph_sizes/np.max(subgraph_sizes)))
        nx.draw(g, pos,
                node_size=subgraph_sizes,
                #node_size=10,
                node_color=subgraph_colors,
                with_labels=False,
                arrowstyle='-',
                width=0.5,
                edge_color='black',
                alpha=0.5,
                cmap=matplotlib.colors.ListedColormap(['tab:blue','tab:red']),
                vmin=0.1,
                vmax=0.5)
    plt.show()

    # Plot barcode distance clustered heatmap
    #sns.set_style("white", {'axes.grid' : False})
    labels = np.array(correction_df['Clade'])
    lut = dict(zip(set(labels), sns.color_palette(cc.glasbey_light, len(set(labels)))))
    row_colors = pd.DataFrame(labels)[0].map(lut).to_numpy()
    cm = sns.clustermap(edit_dist_all,
                        vmin=0, vmax=15, linewidths=0.0, cmap=parula_cm[::-1],
                        yticklabels=0, xticklabels=0, dendrogram_ratio=0.1,
                        row_colors=row_colors, colors_ratio=0.02,
                        figsize=(8,6), cbar_kws={'label': 'Edit Distance'})
    cm.ax_heatmap.axhline(y=0, color='k', linewidth=1)
    cm.ax_heatmap.axhline(y=cm.data.shape[0], color='k', linewidth=1)
    cm.ax_heatmap.axvline(x=0, color='k',linewidth=1)
    cm.ax_heatmap.axvline(x=cm.data.shape[1], color='k', linewidth=1)
    cm.fig.subplots_adjust(right=0.75)
    cm.ax_cbar.set_position((0.8, .7, .01, .2))
    plt.show()

    #  Step 4g: Collapse and count reads for unique TB:CB:UMI barcode sets (after error-correction)
    df2 = df2.groupby(['TracerSeq_corrected','CB_corrected','UMI_raw'],as_index=False).agg({'read_counts': 'sum'}).copy()
    print('Done correcting TB barcodes (step 4/6)')

    # Step 5: Determine the top TB for each transcript

    # Step 5a: Create a new column in the dataframe that contains both the CB:UMI; unique values of this column correspond to individual transcripts
    df2['CB:UMI'] = df2['CB_corrected'] + '-' + df2['UMI_raw']
    transcripts = np.unique(df2['CB:UMI'])
    run_log['# TracerSeq Transcripts Detected'] = len(transcripts)

    # Step 5b: Construct a new dataframe line by line, appending one row for each transcript successfully assigned to a TB
    df3 = pd.DataFrame(columns=list(df2.columns))
    for n, t in enumerate(transcripts):
      # Get the set of all TB sequences associated with this transcript
      flag = df2['CB:UMI'] == t
      # If there is only 1 TB associated with this transcript then we are already done
      if np.sum(flag) == 1:
        df3 = df3.append(df2[flag]) # Assign this TB to this transcript
      # Else, there are multiple TBs associated with this transcript
      else:
        # Use the top TB if it is significantly more abundant than the others
        tmp = df2[flag].sort_values('read_counts', ascending=False).copy()
        if tmp.iloc[0].read_counts > (tmp.iloc[1].read_counts)*10:
          df3 = df3.append(tmp.iloc[0])
    run_log['# TracerSeq Transcripts With Unambiguous TB'] = len(df3)
    del df2

    # Step 5c: Collapse and count UMIs for unique TB:CB sets
    df3.reset_index(drop=True, inplace=True)
    df4 = df3.value_counts(subset=['TracerSeq_corrected', 'CB_corrected'], sort=True).to_frame('UMI_counts').reset_index().copy()
    _,df4['TracerSeq_id'] = np.unique(df4['TracerSeq_corrected'], return_inverse=True)
    print('Done assigning TBs to transcripts (step 5/6)')

    # Step 5d: Write final dataframe to csv, reordering columns to match the convention of Wagner2018 '*TracerCounts.csv' files
    df4 = df4[['CB_corrected','TracerSeq_id','TracerSeq_corrected','UMI_counts']].copy()
    #df4.to_csv(run_params['Save File Path'] + run_params['File Prefix'] + '_CellTracerCounts.csv.gz', index=False, header=False, compression='gzip')
    df4.to_csv(run_params['Save File Path'] + run_params['File Prefix'] + '_CellTracerCounts.csv', index=False, header=False)
    print('Done saving counts csv file (step 6/6)')

    # Print and save run info
    df_params = pd.DataFrame.from_dict(run_params, orient='index', columns=[''], dtype='object')
    df_params.to_csv(run_params['Save File Path'] + run_params['File Prefix'] +'_run_params.csv', index=True, header=False)
    df_log = pd.DataFrame.from_dict(run_log, orient='index', columns=[''], dtype='object')
    df_log.to_csv(run_params['Save File Path'] + run_params['File Prefix'] + '_log.csv', index=True, header=False)

    return df_log

# Define the 'parula' colormap (from Matlab)
parula_cm = [[0.2081, 0.1663, 0.5292],
             [0.2116238095, 0.1897809524, 0.5776761905],
             [0.212252381, 0.2137714286, 0.6269714286],
             [0.2081, 0.2386, 0.6770857143],
             [0.1959047619, 0.2644571429, 0.7279],
             [0.1707285714, 0.2919380952, 0.779247619],
             [0.1252714286, 0.3242428571, 0.8302714286],
             [0.0591333333, 0.3598333333, 0.8683333333],
             [0.0116952381, 0.3875095238, 0.8819571429],
             [0.0059571429, 0.4086142857, 0.8828428571],
             [0.0165142857, 0.4266, 0.8786333333],
             [0.032852381, 0.4430428571, 0.8719571429],
             [0.0498142857, 0.4585714286, 0.8640571429],
             [0.0629333333, 0.4736904762, 0.8554380952],
             [0.0722666667, 0.4886666667, 0.8467],
             [0.0779428571, 0.5039857143, 0.8383714286],
             [0.079347619, 0.5200238095, 0.8311809524],
             [0.0749428571, 0.5375428571, 0.8262714286],
             [0.0640571429, 0.5569857143, 0.8239571429],
             [0.0487714286, 0.5772238095, 0.8228285714],
             [0.0343428571, 0.5965809524, 0.819852381],
             [0.0265, 0.6137, 0.8135],
             [0.0238904762, 0.6286619048, 0.8037619048],
             [0.0230904762, 0.6417857143, 0.7912666667],
             [0.0227714286, 0.6534857143, 0.7767571429],
             [0.0266619048, 0.6641952381, 0.7607190476],
             [0.0383714286, 0.6742714286, 0.743552381],
             [0.0589714286, 0.6837571429, 0.7253857143],
             [0.0843, 0.6928333333, 0.7061666667],
             [0.1132952381, 0.7015, 0.6858571429],
             [0.1452714286, 0.7097571429, 0.6646285714],
             [0.1801333333, 0.7176571429, 0.6424333333],
             [0.2178285714, 0.7250428571, 0.6192619048],
             [0.2586428571, 0.7317142857, 0.5954285714],
             [0.3021714286, 0.7376047619, 0.5711857143],
             [0.3481666667, 0.7424333333, 0.5472666667],
             [0.3952571429, 0.7459, 0.5244428571],
             [0.4420095238, 0.7480809524, 0.5033142857],
             [0.4871238095, 0.7490619048, 0.4839761905],
             [0.5300285714, 0.7491142857, 0.4661142857],
             [0.5708571429, 0.7485190476, 0.4493904762],
             [0.609852381, 0.7473142857, 0.4336857143],
             [0.6473, 0.7456, 0.4188],
             [0.6834190476, 0.7434761905, 0.4044333333],
             [0.7184095238, 0.7411333333, 0.3904761905],
             [0.7524857143, 0.7384, 0.3768142857],
             [0.7858428571, 0.7355666667, 0.3632714286],
             [0.8185047619, 0.7327333333, 0.3497904762],
             [0.8506571429, 0.7299, 0.3360285714],
             [0.8824333333, 0.7274333333, 0.3217],
             [0.9139333333, 0.7257857143, 0.3062761905],
             [0.9449571429, 0.7261142857, 0.2886428571],
             [0.9738952381, 0.7313952381, 0.266647619],
             [0.9937714286, 0.7454571429, 0.240347619],
             [0.9990428571, 0.7653142857, 0.2164142857],
             [0.9955333333, 0.7860571429, 0.196652381],
             [0.988, 0.8066, 0.1793666667],
             [0.9788571429, 0.8271428571, 0.1633142857],
             [0.9697, 0.8481380952, 0.147452381],
             [0.9625857143, 0.8705142857, 0.1309],
             [0.9588714286, 0.8949, 0.1132428571],
             [0.9598238095, 0.9218333333, 0.0948380952],
             [0.9661, 0.9514428571, 0.0755333333],
             [0.9763, 0.9831, 0.0538]]


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


