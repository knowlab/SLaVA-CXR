# SLaVA-CXR: Small Language and Vision Assistant for Chest X-ray Report Automation

**SLaVA-CXR: Small Language and Vision Assistant for Chest X-ray Report Automation** [[Paper](https://arxiv.org/abs/2409.13321)] <br>

## Contents
- [Install](#install)
- [LLaVA-Phi Weights](#llava-weights)
- [Train](#train)
- [Evaluation](#evaluation)

## Environment

```Shell
conda create -n slava_cxr python=3.10 -y
conda activate slava_cxr
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## MODEL
The SLaVA-CXR model can be downloaded in [HuggingFace](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view). 

## Train
The training codes is made available. The training datasets are currently not available.

## Evaluation
Evaluation dataset can be any chest X-ray frontal view image paired with a report.  
We used MIMIC-CXR and IU-Xray datasets in our paper for the evaluation.
We have included IU-Xray questions for impression and findings section automation.
Please download IU-Xray dataset [LINK](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view). 

### Findings Generation
```Shell
CUDA_VISIBLE_DEVICES=0 python -m llava_phi.eval.model_vqa_slava_cxr \
    --model-path ./SLaVA-CXR \
    --question-file iuxray_sample_findings.jsonl \
    --image-folder path_to_iuxray_images \
    --answers-file findings_result.jsonl \
    --conv-mode default \
    --max_new_tokens 512
```
### Impression Summarization
```Shell
CUDA_VISIBLE_DEVICES=0 python -m llava_phi.eval.model_vqa_slava_cxr \
    --model-path ./SLaVA-CXR \
    --question-file iuxray_sample_impression.jsonl \
    --image-folder path_to_iuxray_images \
    --answers-file impression_result.jsonl \
    --conv-mode default \
    --max_new_tokens 256
```
## Citation
```bibtex

@article{wu2024slava,
  title={SLaVA-CXR: Small Language and Vision Assistant for Chest X-ray Report Automation},
  author={Wu, Jinge and Kim, Yunsoo and Shi, Daqian and Cliffton, David and Liu, Fenglin and Wu, Honghan},
  journal={arXiv preprint arXiv:2409.13321},
  year={2024}
}
```

## Acknowledgement
We used the LLaVA-Phi codes to train our model
- [LLaVA-Phi](https://github.com/zhuyiche/llava-phi)
