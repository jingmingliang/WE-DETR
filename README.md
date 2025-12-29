<p align="center">

  <h2 align="center"><strong>WE-DETR: Wavelet-Enhanced Multi-Scale Feature Compensation Detection Transformer for Remote Sensing Small Objects</strong></h2>

<div align="center">
<h5>
<em>Yu Song<sup>1</sup>, Weiguo Zuo<sup>1 </sup>, Ronghua Shang<sup>2</sup>,Haixiao Zhang<sup>3</sup>, Chen Zheng<sup>1</sup>, Zhenzhen Liu<sup>1</sup>,<br/> Xiaohui Yang<sup>1</sup> </em>
    <br><br>
       	<sup>1</sup> Henan University, China, <sup>2</sup> Xidian University, China, <br/> <sup>3</sup> Beijing Institute of Spacecraft Syetem Engineering, China
    </h5>
</div>

# 📚 Contents

- [Abstract](#abstract)

# 📄Abstract
Small object detection in remote sensing imagery remains challenging due to the extremely limited pixel footprint of targets and the severe loss of fine-grained spatial details caused by multi-stage downsampling. To address these issues, we propose WE-DETR, a Wavelet-Enhanced Multi-Scale Feature Compensation Detection Transformer that explicitly preserves and compensates high-frequency information throughout the detection pipeline.
Specifically, WE-DETR introduces a wavelet-enhanced feature compensation backbone that decomposes input imagery into low-frequency and high-frequency sub-bands using discrete wavelet transform, enabling effective preservation of structural details such as edges and textures. A cross-attention fusion module is further designed to facilitate deep interaction between frequency-domain features and spatial-domain semantic features. To enhance multi-scale feature fusion while minimizing information loss, a detail-preserving downsampling strategy based on SPDConv and an efficient CSP-based decomposed large-kernel attention module are incorporated to strengthen shallow-to-deep feature interactions.
Extensive experiments on multiple remote sensing benchmarks, including optical, infrared, and SAR datasets, demonstrate that WE-DETR consistently outperforms state-of-the-art detectors in small-object detection accuracy while maintaining competitive computational efficiency. These results validate the effectiveness and robustness of the proposed wavelet-assisted multi-scale feature compensation framework across diverse sensing modalities. 
