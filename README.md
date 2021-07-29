# Cascaded RGB-D SOD with COME15K dataset (ICCV2021)
This is the official implementaion of CLNet paper "RGB-D Saliency Detection via Cascaded Mutual Information Minimization".

# COME15K RGB-D SOD Dataset
We provide the COME training dataset, which include 8,025 image pairs of RGB-D images for SOD training. The dataset can be found at:
https://drive.google.com/drive/folders/1mGbFKlIJNeW0m7hE-1dGX0b2gcxSMXjB?usp=sharing

We further introduce two sets of testing dataset, namely COME-E and COME-H, which include 4,600 and 3,000 image pairs respectively, and can be downloaded at:
https://drive.google.com/drive/folders/1w0M9YmYBzkMLijy_Blg6RMRshSvPZju-?usp=sharing

# New benchmark
With our new training dataset, we re-train existing RGB-D SOD models, and test on ten benchmark testing dataset, including: SSB (STERE), DES, NLPR, NJU2K, LFSD, SIP, DUT-RGBD, RedWeb-S and our COME-E and COME-H. Please find saliency maps of retained models at (constantly updating):
https://drive.google.com/drive/folders/1lCE8OHeqNdjhE4--yR0FFib2C5DBTgwn?usp=sharing

# Our Bib:

Please cite our paper if necessary:
```
@inproceedings{cascaded_rgbd_sod,
  title={RGB-D Saliency Detection via Cascaded Mutual Information Minimization},
  author={Zhang, Jing and Fan, Deng-Ping and Dai, Yuchao and Yu, Xin and Zhong, Yiran and Barnes, Nick and Shao, Ling},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
# Copyright
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/3.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License</a>.

# Contact

Please drop me an email for further problems or discussion: zjnwpu@gmail.com
