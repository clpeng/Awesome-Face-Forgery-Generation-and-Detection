# Awesome Face Forgery Generation and Detection [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
A curated list of articles and codes related to face forgery generation and detection. 

## Contributing
***

Please feel free to send me pull requests or email (clp.xidian@gmail.com) to update this list together!


## Table of Contents
***

- [Target-specific Face Forgery](#target-specific-face-forgery)        
    - [Face Swap](#face-swap)        
    - [Face Manipulation](#face-manipulation)            
        - [Attribute Manipulation](#attribute-manipulation)            
        - [Expression Reenactment](#expression-reenactment)            
        - [Cross-modality Driven](#cross-modality-driven)    
- [Target-generic Face Forgery](#target-generic-face-forgery)    
- [Face Forgery Detection](#face-forgery-detection)        
    - [Spatial Clue for Detection](#spatial-clue-for-detection)        
    - [Temporal Clue for Detection](#temporal-clue-for-detection)        
    - [Generalizable Detection](#generalizable-forgery-detection)        
    - [Spoofing Detection](#spoofing-forgery-detection)    
- [Databases](#databases)    
- [Survey](#survey)    

## Target-specific Face Forgery
***

### Face Swap
* deepfakes/faceswap (*Github*) [[Code](https://github.com/deepfakes/faceswap)]
* iperov/DeepFaceLab (*Github*) [[Paper](https://arxiv.org/pdf/2005.05535.pdf)] [[Code](https://github.com/iperov/DeepFaceLab)]
* Fast face-swap using convolutional neural networks (*2017 ICCV*) [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Korshunova_Fast_Face-Swap_Using_ICCV_2017_paper.pdf)]
* On face segmentation, face swapping, and face perception (*2018 FG*) [[Paper](https://arxiv.org/abs/1704.06729)] [[Code](https://github.com/YuvalNirkin/face_swap)]
* RSGAN: face swapping and editing using face and hair representation in latent spaces (*2018 arXiv*) [[Paper](https://arxiv.org/abs/1804.03447)]
* FSNet: An identity-aware generative model for image-based face swapping (*2018 ACCV*) [[Paper](https://arxiv.org/abs/1811.12666)]
* Towards open-set identity preserving face synthesis (*2018 CVPR*) [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Bao_Towards_Open-Set_Identity_CVPR_2018_paper.pdf)]
* FSGAN: Subject Agnostic Face Swapping and Reenactment (*2019 ICCV*) [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Nirkin_FSGAN_Subject_Agnostic_Face_Swapping_and_Reenactment_ICCV_2019_paper.pdf)]
* Deepfakes for Medical Video De-Identification: Privacy Protection and Diagnostic Information Preservation (*202003 arXiv*) [[Paper](https://arxiv.org/pdf/2003.00813.pdf)]
* Advancing High Fidelity Identity Swapping for Forgery Detection (*2020 CVPR*) [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Advancing_High_Fidelity_Identity_Swapping_for_Forgery_Detection_CVPR_2020_paper.pdf)] [[arXiv version](https://arxiv.org/abs/1912.13457)]

### Face Manipulation

#### Attribute Manipulation
* Learning residual images for face attribute manipulation (*2017 CVPR*) [[Paper](http://zpascal.net/cvpr2017/Shen_Learning_Residual_Images_CVPR_2017_paper.pdf)] [[Code](https://github.com/MingtaoGuo/Learning-Residual-Images-for-Face-Attribute-Manipulation)]
* Fader networks: Manipulating images by sliding attributes (*2017 NIPS*) [[Paper](https://papers.nips.cc/paper/7178-fader-networksmanipulating-images-by-sliding-attributes.pdf)] [[Code](https://github.com/facebookresearch/FaderNetworks)]
* StarGAN: Unified generative adversarial networks for multi-domain image-to-image translation (*2018 CVPR*) [[Paper](https://zpascal.net/cvpr2018/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf)] [[Code](https://github.com/yunjey/stargan)]
* Facelet-Bank for Fast Portrait Manipulation (*2018 CVPR*) [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Facelet-Bank_for_Fast_CVPR_2018_paper.pdf)] [[Code](https://github.com/yingcong/Facelet_Bank)]
* Glow: Generative flow with invertible 1x1 convolutions (*2018 NIPS*) [[Paper](https://papers.nips.cc/paper/8224-glow-generative-flow-with-invertible-1x1-convolutions.pdf)] [[Code](https://github.com/openai/glow)]
* Mask-aware Photorealistic Face Attribute Manipulation (*2018 arXiv*) [[Paper](https://arxiv.org/abs/1804.08882)]
* Sparsely Grouped Multi-Task Generative Adversarial Networks for Facial Attribute Manipulation (*2018 ACMMM*) [[Paper](https://arxiv.org/abs/1805.07509)] [[Code](https://github.com/zhangqianhui/Sparsely-Grouped-GAN)]
* AttGAN: Facial attribute editing by only changing what you want (*2019 TIP*) [[Paper](http://vipl.ict.ac.cn/uploadfile/upload/2019112511573287.pdf)] [[Code](https://github.com/LynnHo/AttGAN-Tensorflow)]
* STGAN: A unified selective transfer network for arbitrary image attribute editing (*2019 CVPR*) [[Paper](https://zpascal.net/cvpr2019/Liu_STGAN_A_Unified_Selective_Transfer_Network_for_Arbitrary_Image_Attribute_CVPR_2019_paper.pdf)] [[Code](https://github.com/csmliu/STGAN)]
* Semantic component decomposition for face attribute manipulation (*2019 CVPR*) [[Paper](https://zpascal.net/cvpr2019/Chen_Semantic_Component_Decomposition_for_Face_Attribute_Manipulation_CVPR_2019_paper.pdf)] [[Code](https://github.com/yingcong/SemanticComponent)]
* Make a Face: Towards Arbitrary High Fidelity Face Manipulation (*2019 ICCV*) [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_Make_a_Face_Towards_Arbitrary_High_Fidelity_Face_Manipulation_ICCV_2019_paper.pdf)]
* Towards Automatic Face-to-Face Translation (*2019 ACMMM*) [[Paper](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/facetoface_translation/paper.pdf)]
* MulGAN: Facial Attribute Editing by Exemplar (*2019 arXiv*) [[Paper](https://arxiv.org/abs/1912.12396)]
* MaskGAN: Towards Diverse and Interactive Facial Image Manipulation (*2019 arXiv*) [[Paper](https://arxiv.org/abs/1907.11922)] [[Code](https://github.com/switchablenorms/CelebAMask-HQ)]
* PuppetGAN: Cross-Domain Image Manipulation by Demonstration (*2019 ICCV*) [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Usman_PuppetGAN_Cross-Domain_Image_Manipulation_by_Demonstration_ICCV_2019_paper.pdf)]
* StarGAN v2: Diverse Image Synthesis for Multiple Domains (*2020 CVPR*) [[Paper](https://arxiv.org/abs/1912.01865)] [[Code](https://github.com/clovaai/stargan-v2)]
* Fine-Grained Expression Manipulation via Structured Latent Space (*202004 arXiv*) [[Paper](https://arxiv.org/pdf/2004.09769.pdf)]

#### Expression Reenactment
* Real-time expression transfer for facial reenactment (*2015 TOG*) [[Paper](http://www.graphics.stanford.edu/~niessner/papers/2015/10face/thies2015realtime.pdf)]
* Face2face: Real-time face capture and reenactment of RGB videos (*2016 CVPR*) [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Thies_Face2Face_Real-Time_Face_CVPR_2016_paper.pdf)]
* ReenactGAN: Learning to reenact faces via boundary transfer (*2018 ECCV*) [[Paper](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2018_reenactgan.pdf)] [[Code](https://github.com/wywu/ReenactGAN)]
* HeadOn: Real-time Reenactment of Human Portrait Videos (*2018 TOG*) [[Paper](https://arxiv.org/pdf/1805.11729.pdf)]
* Deep video portraits (*2018 TOG*) [[Paper](https://arxiv.org/pdf/1805.11714.pdf)]
* ExprGAN: Facial expression editing with controllable expression intensity (*2018 AAAI*) [[Paper](https://arxiv.org/pdf/1709.03842.pdf)] [[Code](https://github.com/HuiDingUMD/ExprGAN)]
* Geometry guided adversarial facial expression synthesis (*2018 ACMMM*) [[Paper](https://arxiv.org/abs/1712.03474)]
* GANimation: Anatomically-aware facial animation from a single image (*2018 ECCV*) [[Paper](http://www.iri.upc.edu/files/scidoc/2052-GANimation:-Anatomically-aware-Facial-Animation-from-a-Single-Image.pdf)] [[Code](https://github.com/albertpumarola/GANimation)]
* Generating Photorealistic Facial Expressions in Dyadic Interactions (*2018 BMVC*) [[Paper](http://bmvc2018.org/contents/papers/0590.pdf)]
* Dynamic Facial Expression Generation on Hilbert Hypersphere with Conditional Wasserstein Generative Adversarial Nets (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1907.10087.pdf)]
* 3D guided fine-grained face manipulation (*2019 CVPR*) [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Geng_3D_Guided_Fine-Grained_Face_Manipulation_CVPR_2019_paper.pdf)]
* Few-shot adversarial learning of realistic neural talking head models (*2019 ICCV*) [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zakharov_Few-Shot_Adversarial_Learning_of_Realistic_Neural_Talking_Head_Models_ICCV_2019_paper.pdf)] [[Code1](https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models)] [[Code2](https://github.com/grey-eye/talking-heads)] [[Code3](https://github.com/shoutOutYangJie/Few-Shot-Adversarial-Learning-for-face-swap)]
* Deferred Neural Rendering: Image Synthesis using Neural Textures (*2019 TOG*) [[Paper](https://arxiv.org/pdf/1904.12356.pdf)] [[Code](https://github.com/SSRSGJYD/NeuralTexture)]
* MarioNETte: Few-shot Face Reenactment Preserving Identity of Unseen Targets (*2019 arXiv*) [[Paper](https://arxiv.org/abs/1911.08139)]
* Unconstrained Facial Expression Transfer using Style-based Generator (*2019 arXiv*) [[Paper](https://arxiv.org/abs/1912.06253)]
* ICface: Interpretable and Controllable Face Reenactment Using GANs (*2020 WACV*) [[Paper](https://arxiv.org/pdf/1904.01909.pdf)] [[Code](https://github.com/Blade6570/icface)]
* Realistic Face Reenactment via Self-Supervised Disentangling of Identity and Pose (*202003 arXiv*) [[Paper](https://arxiv.org/pdf/2003.12957.pdf)]
* APB2Face: Audio-guided face reenactment with auxiliary pose and blink signals (*202004 arXiv*) [[Paper](https://arxiv.org/pdf/2004.14569.pdf)]
* One-Shot Identity-Preserving Portrait Reenactment (*202004 arXiv*) [[Paper](https://arxiv.org/pdf/2004.12452.pdf)]
* FReeNet: Multi-Identity Face Reenactment (*2020 CVPR*) [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_FReeNet_Multi-Identity_Face_Reenactment_CVPR_2020_paper.pdf)] [[Code](https://github.com/zhangzjn/FReeNet)]
* Learning Identity-Invariant Motion Representations for Cross-ID Face Reenactment (*2020 CVPR*) [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Learning_Identity-Invariant_Motion_Representations_for_Cross-ID_Face_Reenactment_CVPR_2020_paper.pdf)]
* FaR-GAN for One-Shot Face Reenactment (*202005 arXiv*) [[Paper](https://arxiv.org/pdf/2005.06402.pdf)]
* ReenactNet: Real-time Full Head Reenactment (*202006 arXiv*) [[Paper](https://arxiv.org/pdf/2006.10500.pdf)]



#### Cross-modality Manipulation
* Synthesizing Obama: learning lip sync from audio (*2017 TOG*) [[Paper](https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf)] [[Code](https://github.com/supasorn/synthesizing_obama_network_training)]
* Face synthesis from visual attributes via sketch using conditional VAEsand GANs (*2017 arXiv*) [[Paper](https://arxiv.org/abs/1801.00077)]
* GP-GAN: Gender Preserving GAN for Synthesizing Faces from Landmarks (*2018 ICPR*) [[Paper](https://engineering.jhu.edu/vpatel36/wp-content/uploads/2018/08/GPGAN_icpr18_camera_ready.pdf)]
* Speech2Face: Learning the Face Behind a Voice (*2019 CVPR*) [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Oh_Speech2Face_Learning_the_Face_Behind_a_Voice_CVPR_2019_paper.pdf)]
* Face Reconstruction from Voice using Generative Adversarial Networks (*2019 NIPS*) [[Paper](https://papers.nips.cc/paper/8768-face-reconstruction-from-voice-using-generative-adversarial-networks.pdf)]
* Neural Voice Puppetry: Audio-driven Facial Reenactment (*2019 arXiv*) [[Paper](https://arxiv.org/abs/1912.05566)]
* Realistic Speech-Driven Facial Animation with GANs (*2019 IJCV*) [[Paper](https://arxiv.org/abs/1906.06337)]
* You said that?: Synthesising talking faces from audio (*2019 IJCV*) [[Paper](https://www.robots.ox.ac.uk/~vgg/publications/2019/Jamaludin19/jamaludin19.pdf)]
* Text-based editing of talking-head video (*2019 TOG*) [[Paper](https://www.ohadf.com/projects/text-based-editing/data/text-based-editing.pdf)]
* FTGAN: A Fully-trained Generative Adversarial Networks for Text to Face Generation (*2019 arXiv*) [[Paper](https://arxiv.org/abs/1904.05729)]
* Everybody’s Talkin’: Let Me Talk as You Want (*202001 arXiv*) [[Paper](https://arxiv.org/abs/2001.05201)]
* Identity-Preserving Realistic Talking Face Generation (*202005 arXiv*) [[Paper](https://arxiv.org/pdf/2005.12318.pdf)]
* Talking-head Generation with Rhythmic Head Motion (*2020 ECCV*) [[Paper](https://arxiv.org/pdf/2007.08547.pdf)] [[Code](https://github.com/lelechen63/Talking-head-Generation-with-Rhythmic-Head-Motion)]

## Target-generic Face Forgery (Representative)
***
* Generative Adversarial Nets (*2014 NIPS*) [[Paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)] [[Code](https://github.com/goodfeli/adversarial)]
* Unsupervised representation learning with deep convolutional generative adversarial networks (*2016 ICLR*) [[Paper](https://arxiv.org/pdf/1511.06434.pdf)] [[Code](https://github.com/Newmu/dcgan_code)]
* Progressive growing of GANs for improved quality, stability, and variation (*2018 ICLR*) [[Paper](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf)] [[Code](https://github.com/tkarras/progressive_growing_of_gans)]
* Spectral normalization for generative adversarial networks (*2018 ICLR*) [[Paper](https://openreview.net/pdf?id=B1QRgziT-)] [[Code](https://github.com/pfnet-research/sngan_projection)] [[Code](https://github.com/godisboy/SN-GAN)]
* Self-attention generative adversarial networks (*2018 arXiv*) [[Paper](https://arxiv.org/pdf/1805.08318.pdf)] [[Code](https://github.com/heykeetae/Self-Attention-GAN)]
* A Style-Based Generator Architecture for Generative Adversarial Networks (*2019 CVPR*) [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf)] [[Code](https://github.com/NVlabs/stylegan)]
* Large Scale GAN Training for High Fidelity Natural Image Synthesis (*2019 ICLR*) [[Paper](https://arxiv.org/pdf/1809.11096.pdf)] [[Code1](https://github.com/AaronLeong/BigGAN-pytorch)] [[Code2](https://github.com/taki0112/BigGAN-Tensorflow)]
* Analyzing and improving the image quality of StyleGAN (*2019 arXiv*) [[Paper](https://arxiv.org/abs/1912.04958)] [[Code1](https://github.com/NVlabs/stylegan2)] [[Code2](https://github.com/rosinality/stylegan2-pytorch)]
* One-Shot Domain Adaptation For Face Generation (*202003 arXiv*) [[Paper](https://arxiv.org/pdf/2003.12869.pdf)]

## Face Forgery Detection
***

### Spatial Clue for Detection
* Automated face swapping and its detection (*2017 ICSIP*) [[Paper](https://ieeexplore.ieee.org/document/8124497)]
* Two-stream neural networks for tampered face detection (*2017 CVPRW*) [[Paper](https://arxiv.org/abs/1803.11276)]
* Can Forensic Detectors Identify GAN Generated Images? (*2018 APSIPA*) [[Paper](http://www.apsipa.org/proceedings/2018/pdfs/0000722.pdf)]
* Deepfakes: a new threat to face recognition? assessment and detection (*2018 arXiv*) [[Paper](https://arxiv.org/abs/1812.08685)]
* Detection of Deep Network Generated Images Using Disparities in Color Components (*2018 arXiv*) [[Paper](https://arxiv.org/pdf/1808.07276.pdf)] [[Code](https://github.com/lihaod/GAN_image_detection)]
* Fake Faces Identification via Convolutional Neural Network (*2018 IH&MMSec*) [[Paper](https://dl.acm.org/doi/10.1145/3206004.3206009)]
* Learning to detect fake face images in the wild (*2018 IS3C*) [[Paper](https://arxiv.org/ftp/arxiv/papers/1809/1809.08754.pdf)] [[Code](https://github.com/jesse1029/Fake-Face-Images-Detection-Tensorflow)]
* Detecting Both Machine and Human Created Fake Face Images In the Wild (*2018 MPS*) [[Paper](https://dl.acm.org/doi/10.1145/3267357.3267367)]
* Detection of Deepfake Video Manipulation (*2018 IMVIP*) [[Paper](https://www.researchgate.net/publication/329814168_Detection_of_Deepfake_Video_Manipulation)]
* Secure detection of image manipulation by means of random feature selection (*2019 TIFS*) [[Paper](https://arxiv.org/pdf/1802.00573.pdf)]
* Exploiting Human Social Cognition for the Detection of Fake and Fraudulent Faces via Memory Networks (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1911.07844.pdf)]
* Swapped Face Detection using Deep Learning and Subjective Assessment (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1909.04217.pdf)]
* Detection of Fake Images Via The Ensemble of Deep Representations from Multi Color Spaces (*2019 ICIP*) [[Paper](https://ieeexplore.ieee.org/abstract/document/8803740/)]
* Detection GAN-Generated Imagery Using Saturation Cues (*2019 ICIP*) [[Paper](https://ieeexplore.ieee.org/document/8803661)]
* Detecting GAN generated fake images using co-occurrence matrices (*2019 Electronic Imaging*) [[Paper](https://arxiv.org/pdf/1903.06836.pdf)]
* Exposing DeepFake Videos By Detecting Face Warping Artifacts (*2019 CVPRW*) [[Paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Li_Exposing_DeepFake_Videos_By_Detecting_Face_Warping_Artifacts_CVPRW_2019_paper.pdf)] [[Code](https://github.com/danmohaha/CVPRW2019_Face_Artifacts)]
* Exposing GAN-synthesized Faces Using Landmark Locations (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1904.00167.pdf)]
* Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations (*2019 WACVW*) [[Paper](https://ieeexplore.ieee.org/document/8638330)] [[Code](https://github.com/FalkoMatern/Exploiting-Visual-Artifacts)]
* Detecting and Simulating Artifacts in GAN Fake Images (*2019 WIFS*) [[Paper](https://arxiv.org/pdf/1907.06515.pdf)] [[Code](https://github.com/ColumbiaDVMM/AutoGAN)]
* On the detection of digital face manipulation (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1910.01717.pdf)]
* On the generalization of GAN image forensics (*2019 CCBR*) [[Paper](https://arxiv.org/pdf/1902.11153.pdf)]
* Unmasking DeepFakes with simple Features (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1911.00686.pdf)] [[Code](https://github.com/cc-hpc-itwm/DeepFakeDetection)]
* Face image manipulation detection based on a convolutional neural network (*2019 ESWA*) [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417419302350)]
* Do GANs leave artificial fingerprints? (*2019 MIPR*) [[Paper](https://arxiv.org/pdf/1812.11842.pdf)]
* Attributing fake images to GANs: Learning and analyzing GAN fingerprints (*2019 ICCV*) [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Attributing_Fake_Images_to_GANs_Learning_and_Analyzing_GAN_Fingerprints_ICCV_2019_paper.pdf)] [[Code](https://github.com/ningyu1991/GANFingerprints)]
* Multi-task learning for detecting and segmenting manipulated facial images and videos (*2019 BTAS*) [[Paper](https://arxiv.org/pdf/1906.06876.pdf)] [[Code](https://github.com/nii-yamagishilab/ClassNSeg)]
* Poster: Towards Robust Open-World Detection of Deepfakes (*2019 CCS*) [[Paper](https://dl.acm.org/doi/abs/10.1145/3319535.3363269)]
* Extracting deep local features to detect manipulated images of human faces (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1911.13269.pdf)]
* Zooming into Face Forensics: A Pixel-level Analysis (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1912.05790.pdf)]
* Fakespotter: A simple baseline for spotting ai-synthesized fake faces (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1909.06122.pdf)]
* Capsule-forensics: Using capsule networks to detect forged images and videos (*2019 ICASSP*) [[Paper](https://arxiv.org/pdf/1810.11215.pdf)] [[Code](https://github.com/nii-yamagishilab/Capsule-Forensics)]
* Use of a Capsule Network to Detect Fake Images and Videos (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1910.12467.pdf)] [[Code](https://github.com/nii-yamagishilab/Capsule-Forensics-v2)]
* Deep Fake Image Detection based on Pairwise Learning (*2020 Applied Science*) [[Paper](https://www.researchgate.net/publication/338382561_Deep_Fake_Image_Detection_Based_on_Pairwise_Learning)]
* Detecting Face2Face Facial Reenactment in Videos (*2020 WACV*) [[Paper](http://openaccess.thecvf.com/content_WACV_2020/papers/Kumar_Detecting_Face2Face_Facial_Reenactment_in_Videos_WACV_2020_paper.pdf)]
* FakeLocator: Robust Localization of GAN-Based Face Manipulations via Semantic Segmentation Networks with Bells and Whistles (*2020 arXiv*) [[Paper](https://arxiv.org/pdf/2001.09598.pdf)]
* FDFtNet: Facing Off Fake Images using Fake Detection Fine-tuning Network (*2020 arXiv*) [[Paper](https://arxiv.org/pdf/2001.01265.pdf)] [[Code](https://github.com/cutz-j/FDFtNet)]
* Global Texture Enhancement for Fake Face Detection in the Wild (*2020 arXiv*) [[Paper](https://arxiv.org/pdf/2002.00133.pdf)]
* Detecting Deepfakes with Metric Learning (*202003 arXiv*) [[Paper](https://arxiv.org/pdf/2003.08645.pdf)]
* Fake Generated Painting Detection via Frequency Analysis (*202003 arXiv*) [[Paper](https://arxiv.org/pdf/2003.02467.pdf)]
* Leveraging Frequency Analysis for Deep Fake Image Recognition (*202003 arXiv*) [[Paper](https://arxiv.org/pdf/2003.08685.pdf)] [[Code](https://github.com/RUB-SysSec/GANDCTAnalysis)]
* One-Shot GAN Generated Fake Face Detection (*202003 arXiv*) [[Paper](https://arxiv.org/pdf/2003.08685.pdf)]
* DeepFake Detection by Analyzing Convolutional Traces (*2020 CVPRW*) [[Paper](https://arxiv.org/pdf/2004.10448.pdf)] [[Website](https://iplab.dmi.unict.it/mfs/DeepFake/)]
* DeepFakes Evolution: Analysis of Facial Regions and Fake Detection Performance (*202004 arXiv*) [[Paper](https://arxiv.org/pdf/2004.07532.pdf)]
* On the use of Benford's law to detect GAN-generated images (*202004 arXiv*) [[Paper](https://arxiv.org/pdf/2004.07682.pdf)]
* Video Face Manipulation Detection Through Ensemble of CNNs (*202004 arXiv*) [[Paper](https://arxiv.org/pdf/2004.07676.pdf)] [[Code](https://github.com/polimi-ispl/icpr2020dfdc)]
* Detecting Forged Facial Videos using convolutional neural network (*202005 arXiv*) [[Paper](https://arxiv.org/pdf/2005.08344.pdf)]
* Fake Face Detection via Adaptive Residuals Extraction Network (*202005 arXiv*) [[Paper](https://arxiv.org/pdf/2005.04945.pdf)]
* Manipulated Face Detector: Joint Spatial and Frequency Domain Attention Network (*202005 arXiv*) [[Paper](https://arxiv.org/pdf/2005.02958.pdf)]
* A Face Preprocessing Approach for Improved DeepFake Detection (*202006 arXiv*) [[Paper](https://arxiv.org/pdf/2006.07084.pdf)]
* A Note on Deepfake Detection with Low-Resources (*202006 arXiv*) [[Paper](https://arxiv.org/pdf/2006.05183.pdf)]
* Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues (*2020 ECCV*) [[Paper](https://arxiv.org/pdf/2007.09355.pdf)]
* CNN Detection of GAN-Generated Face Images based on Cross-Band Co-occurrences Analysis (*2020 WIFS*) [[Paper](https://arxiv.org/pdf/2007.12909.pdf)]
* Detection, Attribution and Localization of GAN Generated Images (*202007 arXiv*) [[Paper](https://arxiv.org/pdf/2007.10466.pdf)]


### Temporal Clue for Detection
* Mesonet: a compact facial video forgery detection network (*2018 WIFS*) [[Paper](https://arxiv.org/pdf/1809.00888.pdf)] [[Code](https://github.com/DariusAf/MesoNet)]
* In Ictu Oculi: Exposing AI created fake videos by detecting eye blinking (*2018 WIFS*) [[Paper](https://arxiv.org/pdf/1806.02877.pdf)] [[Code](https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi)]
* Deepfake Video Detection Using Recurrent Neural Networks (*2018 AVSS*) [[Paper](https://gangw.cs.illinois.edu/class/cs598/papers/AVSS18-deepfake.pdf)]
* Exposing deep fakes using inconsistent head poses (*2019 ICASSP*) [[Paper](https://arxiv.org/pdf/1811.00661.pdf)]
* Protecting world leaders against deep fakes (*2019 CVPRW*) [[Paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.pdf)]
* FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1901.02212.pdf)]
* Recurrent Convolutional Strategies for Face Manipulation Detection in Videos (*2019 CVPRW*) [[Paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Sabir_Recurrent_Convolutional_Strategies_for_Face_Manipulation_Detection_in_Videos_CVPRW_2019_paper.pdf)]
* Predicting Heart Rate Variations of Deepfake Videos using Neural ODE (*2019 ICCVW*) [[Paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CVPM/Fernandes_Predicting_Heart_Rate_Variations_of_Deepfake_Videos_using_Neural_ODE_ICCVW_2019_paper.pdf)]
* Deepfake Video Detection through Optical Flow based CNN (*2019 ICCVW*) [[Paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/HBU/Amerini_Deepfake_Video_Detection_through_Optical_Flow_Based_CNN_ICCVW_2019_paper.pdf)]
* Emotions Don't Lie: A Deepfake Detection Method using Audio-Visual Affective Cues (*202003 arXiv*) [[Paper](https://arxiv.org/pdf/2003.06711.pdf)]
* Deep Face Forgery Detection (*202004 arXiv*) [[Paper](https://arxiv.org/pdf/2004.11804.pdf)]
* Deepfakes Detection with Automatic Face Weighting (*2020 CVPRW*) [[Paper](https://arxiv.org/pdf/2004.12027.pdf)]
* Detecting Deep-Fake Videos from Appearance and Behavior (*202004 arXiv*) [[Paper](https://arxiv.org/pdf/2004.14491.pdf)]
* Detecting Deep-Fake Videos from Phoneme-Viseme Mismatches (*2020 CVPRW*) [[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Agarwal_Detecting_Deep-Fake_Videos_From_Phoneme-Viseme_Mismatches_CVPRW_2020_paper.pdf)]
* Towards Untrusted Social Video Verification to Combat Deepfakes
via Face Geometry Consistency (*2020 CVPRW*) [[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Tursman_Towards_Untrusted_Social_Video_Verification_to_Combat_Deepfakes_via_Face_CVPRW_2020_paper.pdf)] [[Code](https://github.com/brownvc/social-video-verification)]
* Not made for each other– Audio-Visual Dissonance-based Deepfake Detection and Localization (*202005 arXiv*) [[Paper](https://arxiv.org/pdf/2005.14405.pdf)]
* DeepRhythm: Exposing DeepFakes with Attentional Visual Heartbeat Rhythms (*202006 arXiv*) [[Paper](https://arxiv.org/pdf/2006.07634.pdf)]
* Deepfake Detection using Spatiotemporal Convolutional Networks (*202006 arXiv*) [[Paper](https://arxiv.org/pdf/2006.14749.pdf)]
* Interpretable Deepfake Detection via Dynamic Prototypes (*202006 arXiv*) [[Paper](https://arxiv.org/pdf/2006.15473.pdf)]
* Dynamic texture analysis for detecting fake faces in video sequences (*202007 arXiv*) [[Paper](https://arxiv.org/pdf/2007.15271.pdf)]
* Detecting Deepfake Videos: An Analysis of Three Techniques (*202007 arXiv*) [[Paper](https://arxiv.org/pdf/2007.08517.pdf)]


### Generalizable Forgery Detection
* ForensicTransfer: Weakly-supervised domain adaptation for forgery detection (*2018 arXiv*) [[Paper](https://arxiv.org/pdf/1812.02510.pdf)]
* Towards generalizable forgery detection with locality-aware autoencoder (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1909.05999.pdf)]
* Incremental learning for the detection and classification of GAN-generated images (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1910.01568.pdf)]
* CNN-generated images are surprisingly easy to spot... for now (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1912.11035.pdf)] [[Code](https://github.com/PeterWang512/CNNDetection)]
* Face X-ray for More General Face Forgery Detection (*2020 CVPR*) [[Paper](https://arxiv.org/pdf/1912.13458.pdf)]
* Detecting CNN-Generated Facial Images in Real-World Scenarios (*2020 CVPRW*) [[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Hulzebosch_Detecting_CNN-Generated_Facial_Images_in_Real-World_Scenarios_CVPRW_2020_paper.pdf)]
* OC-FakeDect: Classifying Deepfakes Using One-class Variational Autoencoder (*2020 CVPRW*) [[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Khalid_OC-FakeDect_Classifying_Deepfakes_Using_One-Class_Variational_Autoencoder_CVPRW_2020_paper.pdf)]


### Spoofing Forgery Detection
* Security of Facial Forensics Models Against Adversarial Attacks (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1911.00660.pdf)]
* Real or Fake? Spoofing State-Of-The-Art Face Synthesis Detection Systems (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1911.05351.pdf)]
* Adversarial Perturbations Fool Deepfake Detectors (*2020 IJCNN*) [[Paper](https://arxiv.org/pdf/2003.10596.pdf)] [[Code](https://github.com/ApGa/adversarial_deepfakes)]
* Disrupting DeepFakes: Adversarial Attacks Against Conditional Image Translation Networks and Facial Manipulation Systems (*202003 arXiv*) [[Paper](https://arxiv.org/pdf/2003.01279.pdf)] [[Code](https://github.com/natanielruiz/disrupting-deepfakes)]
* Evading Deepfake-Image Detectors with White- and Black-Box Attacks (*202004 arXiv*) [[Paper](https://arxiv.org/pdf/2004.00622.pdf)]
* Defending against GAN-based Deepfake Attacks via
Transformation-aware Adversarial Faces (*202006 arXiv*) [[Paper](https://arxiv.org/pdf/2006.07421.pdf)]
* Disrupting Deepfakes with an Adversarial Attack
that Survives Training (*202006 arXiv*) [[Paper](https://arxiv.org/pdf/2006.12247.pdf)]
* FakePolisher: Making DeepFakes More Detection-Evasive by
Shallow Reconstruction (*202006 arXiv*) [[Paper](https://arxiv.org/pdf/2006.07533.pdf)]
* Protecting Against Image Translation Deepfakes by Leaking Universal Perturbations from Black-Box Neural Networks (*202006 arXiv*) [[Paper](https://arxiv.org/pdf/2006.06493.pdf)]


## Others
* Detecting Video Speed Manipulation (*2020 CVPRW*) [[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Hosler_Detecting_Video_Speed_Manipulation_CVPRW_2020_paper.pdf)]
* The eyes know it: FakeET- An Eye-tracking Database to Understand Deepfake Perception (*202006 arXiv*) [[Paper](https://arxiv.org/pdf/2006.06961.pdf)]


## Databases
***
* [*FFW*] Fake Face Detection Methods: Can They Be Generalized? (*2018 BIOSIG*) [[Paper](http://ali.khodabakhsh.org/wp-content/uploads/Publications/BIOSIG_2018_Fake%20Face%20Detection%20Methods%20Can%20They%20Be%20Generalized.pdf)] [[Download](http://ali.khodabakhsh.org/research/ffw/)]
* [*UADFV*] In Ictu Oculi: Exposing AI created fake videos by detecting eye blinking (*2018 WIFS*) [[Paper](https://arxiv.org/pdf/1806.02877.pdf)] [[Download](https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi)]
* [*DeepfakeTIMIT*] Deepfakes: a new threat to face recognition? assessment and detection (*2018 arXiv*) [[Paper](https://arxiv.org/pdf/1812.08685.pdf)] [[Download](https://www.idiap.ch/dataset/deepfaketimit)]
* [*FaceForensics++ & DFD*] FaceForensics++: Learning to Detect Manipulated Facial Images (*2019 ICCV*) [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf)] [[Download](https://github.com/ondyari/FaceForensics)]
* [*Celeb-DF*] Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics (*2020 CVPR*) [[Paper](https://arxiv.org/pdf/1909.12962.pdf)] [[Download](https://github.com/danmohaha/celeb-deepfakeforensics)]
* [*DFFD (Diverse Fake Face Dataset)*] On the detection of digital face manipulation (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1910.01717.pdf)]
* [*DFDC (Deepfake Detection Challenge)*] The Deepfake Detection Challenge (DFDC) Preview Dataset (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1910.08854.pdf)] [[Download](https://deepfakedetectionchallenge.ai/)]
* [*DeeperForensics-1.0*] DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection (*2020 arXiv*) [[Paper](https://arxiv.org/pdf/2001.03024.pdf)] [[Download](https://github.com/EndlessSora/DeeperForensics-1.0)]


## Survey
***
* Deep Learning for Deepfakes Creation and Detection (*2019 arXiv*) [[Paper](https://arxiv.org/pdf/1909.11573.pdf)]
* DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection (*2020 arXiv*) [[Paper](https://arxiv.org/pdf/2001.00179.pdf)]
* Media Forensics and DeepFakes: an Overview (*2020 arXiv*) [[Paper](https://arxiv.org/pdf/2001.06564.pdf)]
* DeepFake Detection: Current Challenges and Next Steps (*202003 arXiv*) [[Paper](https://arxiv.org/pdf/2003.09234.pdf)]
* The Creation and Detection of Deepfakes: A Survey (*202004 arXiv*) [[Paper](https://arxiv.org/pdf/2004.11138.pdf)]

## Related Links
***

* [datamllab/awesome-deepfakes-materials](https://github.com/datamllab/awesome-deepfakes-materials)
* [Qingcsai/awesome-Deepfakes](https://github.com/Qingcsai/awesome-Deepfakes)
* [drimpossible/awesome-deepfake-detection](https://github.com/drimpossible/awesome-deepfake-detection)
* [subinium/awesome-deepfake-porn-detection](https://github.com/subinium/awesome-deepfake-porn-detection)
* [aerophile/awesome-deepfakes](https://github.com/aerophile/awesome-deepfakes)
* [592McAvoy/fake-face-detection](https://github.com/592McAvoy/fake-face-detection)

## License
***

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)
