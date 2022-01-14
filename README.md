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
