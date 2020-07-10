## Welcome

Since December 2019, the novel COVID-19 virus has caused a global pandemic and infected millions of people across 200 countries. 
A key step in controlling the infection is that of identifying infected people. 
In addition to the Reverse Transcription Polymerase Chain Reaction (RT-PCR) tests, lung CT scan analysis has emerged as another essential testing method. 


[Monash Machine Vision Group](http://www.mmvg.org/) recently constructed a COVID-19 CT Reports (COV-CTR) dataset and proposed a new medical report generation approach via AI technologies to describe lung CT images.

### COV-CTR Dataset

We invited three radiologists come from **The First Affiliated Hospital of Harbin Medical University** with more than five years of working experience to apply their diagnostic skills to the public [COVID-CT Dataset](https://github.com/UCSD-AI4H/COVID-CT) and use this information to construct the COV-CTR dataset. 
The COV-CTR consists 728 images (349 for COVID-19 and 379 for Non-COVID) collected from published papers and their corresponding paired Chinese reports. 
More information about these images could be found on [COVID-CT Dataset](https://github.com/UCSD-AI4H/COVID-CT).

The **COV-CTR** including all images and paired Chinese reports is available online, 
[DownLoad Now](https://github.com/Draven-Lee/COVID-19-CT-Reports-Generation/blob/master/Datasets).

![sample]({{ "/imgs/2019-novel-Coronavirus-severe-adult-respiratory-dist_2020_International-Jour-p3-89%250.jpg" | prepend: site.baseurl }})
<img src="imgs/2019-novel-Coronavirus-severe-adult-respiratory-dist_2020_International-Jour-p3-89%250.jpg" width="800" />

A sample from COV-CTR whose report is 
**The thorax was symmetrical, the mediastinal shadow was in the middle, 
and no enlarged lymph nodes were found in the mediastinum. Bilateral lung texture clear, 
the right upper lobe of the pleura can be seen small patches and lamellar ground glass shadow, edge unclear. 
The bronchi of the lobes were unobstructed, and there was no abnormal density shadow in the thoracic cavity on both sides.**


### Auxiliary Signal-Guided Knowledge Encoder-Decoder for Medical Report Generation

Beyond the common difficulties faced in the natural image captioning, 
medical report generation specifically requires the model to describe a medical 
image with a fine-grained and semantic-coherence paragraph that should satisfy both medical commonsense and logic. 
Previous works generally extract the global image features and attempt to generate a paragraph that is similar to referenced reports; 
however, this approach has two limitations. Firstly, the regions of primary interest to radiologists are usually located in a small area of the global image, 
meaning that the remainder parts of the image could be considered as irrelevant noise in the training procedure. 
Secondly, there are many similar sentences used in each medical report to describe the normal regions of the image, 
which causes serious data bias. This deviation is likely to teach models to generate these inessential sentences on a regular basis. 
To address these problems, we propose an Auxiliary Signal-Guided Knowledge Encoder-Decoder (ASGK) to mimic radiologists' working patterns. 
In more detail, ASGK integrates internal visual feature fusion and external medical linguistic information to guide medical knowledge transfer and learning. 
The core structure of ASGK consists of a medical graph encoder and a natural language decoder, inspired by advanced Generative Pre-Training (GPT).


![ov]({{ "/imgs/ov.jpg" | prepend: site.baseurl }})
![ov](https://github.com/Draven-Lee/COVCTR/blob/master/imgs/ov.jpg)

An overview of our ASGK approach. The ASGK model consists of a medical graph encoder and a natural language decoder. T and MCS represent threshold and max connection select operation respectively.

![results]({{ "/imgs/results.jpg" | prepend: site.baseurl }})

Sample output of our approach on both CX-CHR and COV-CTR datasets. In the medical tag graphs, we show the nodes whose value (which is equal to the classification probability) 
exceeds 0.5 and edges whose weights are more than 0.3. To read the image clearly, we show the values of some edges in the appropriate places. 
The underlined text indicates alignment between ground truth reports and generated reports.


The initial findings are summarised and submitted to the 34th NIPS which are also available in [Arxiv](https://arxiv.org/abs/2006.03744).

If you are interested in this dataset and would like to jointly augment this dataset, please connect us: Mingjie.Li@monash.edu


If you use this dataset, please cite our [work](https://arxiv.org/abs/2006.03744).
