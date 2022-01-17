# # DA-for-SBR

Data augmentation method for session-based recommendation


item : <img src="https://latex.codecogs.com/svg.image?v" title="v" />\
FA = Frequency of appearance\
HS = Highest similarity value

### method1
<img src="https://latex.codecogs.com/svg.image?s=\alpha&space;\mathcal{N}(\log_2FA|\mu_1,\sigma_1)&plus;\beta\mathcal{N}(HS|\mu_2,\sigma_2)" title="s=\alpha \mathcal{N}(\log_2FA|\mu_1,\sigma_1)+\beta\mathcal{N}(HS|\mu_2,\sigma_2)" />

### method2
<img src="https://latex.codecogs.com/svg.image?s=\alpha\frac{HS}{\sqrt{FA}}" title="s=\alpha\frac{HS}{\sqrt{FA}}" />

### method3
<img src="https://latex.codecogs.com/svg.image?s&space;=&space;\alpha\frac{HS}{FA}" title="s = \alpha\frac{HS}{FA}" />


---
## Start

by running preprocessing.py, following folder hierarchy is made.

da_for_sbr(main folder)\
&emsp;&emsp;|-main.ipynb\
&emsp;&emsp;|-preprocessing.py\
&emsp;&emsp;|-simmetric.py\
&emsp;&emsp;|-utils.py\
&emsp;&emsp;|-narm_torch\
&emsp;&emsp;|-srgnn_torch\
&emsp;&emsp;|-exps\
&emsp;&emsp;|&emsp;&emsp;|-experiment1\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-**yoochoose** (train files, test files)\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-**diginetica** (train files, test files)\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-**result_narm_yoochoose**\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-y064\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-y128\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-y256\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-y512\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-**result_narm_diginetica**\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-d001\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-d003\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-d006\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-d012\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-**result_srgnn_yoochoose**\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-y064\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-y128\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-y256\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-y512\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-**result_srgnn_diginetica**\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-d001\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-d003\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-d006\
&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|-d012\
&emsp;&emsp;|&emsp;&emsp;|-experiment(n)\
&emsp;&emsp;|&emsp;&emsp;|-train_item_views.csv\
&emsp;&emsp;|&emsp;&emsp;|-yoochoose-clicks-withHeader.dat\


### data download
-train_item_views.csv (https://drive.google.com/file/d/14_Gej2IzMyIR6bVQ0O7mR34Ln20NYt4v/view?usp=sharing)

-yoochoose-clicks-withHeader.dat (https://drive.google.com/file/d/14YJ6Pntx3a2b9If9DgzF9Ax2Po9SUk_W/view?usp=sharing)
