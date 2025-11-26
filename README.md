# Evaluating EEG-to-Text Models through Noise-Based Performance Analysis
📄 Official codebase for our journal paper:  
**Jo et al., “Evaluating EEG-to-text models through noise-based performance analysis”, Scientific Reports, 2025.**

---

## 🚀 Overview
This repository provides the implementation and evaluation pipeline used in our published work on EEG-to-Text decoding reliability. We introduce a **noise-baseline diagnostic methodology** to rigorously assess whether neural decoding models truly learn EEG representations—or simply memorize linguistic patterns.

This work significantly extends our earlier findings from:
- **"Are EEG-to-Text Models Working?"** (IJCAI Workshop 2024)  
- **Correction and analysis of AAAI 2022 model evaluation protocols**

---

## 🧪 Key Findings
- Current EEG-to-Text models exhibit **similar or better performance on noise vs. EEG**
- Teacher-forcing evaluations lead to **>3× performance inflation**
- Models rely heavily on **pretrained language priors** rather than genuine neural decoding
- We establish **mandatory evaluation standards** for future EEG-to-Text studies

📘 Full paper (open access):  
> https://doi.org/10.1038/s41598-025-29587-x

📚 ArXiv preprint (extended technical content):  
> https://arxiv.org/abs/2405.06459

---

## 📂 Branch Structure
| Branch | Description |
|--------|-------------|
| `main` | Official implementation with noise-baseline evaluation & journal-published experiments |
| `legacy` | Original reproduction and correction of the AAAI 2022 EEG-to-Text baseline |

---

## 🔍 Research Background & Contribution
We reproduce and correct the model from:  
> Wang & Ji — *AAAI 2022, Open Vocabulary EEG-to-Text Decoding*

### ❗Teacher-forcing issue identified
The original evaluation inadvertently used **implicit teacher-forcing**, resulting in unrealistic predictions (string generation depended on ground-truth tokens).

We provide:
- A corrected **model.generate()**-based inference procedure
- Re-evaluation across **BART, T5, Pegasus** architectures
- Extensive **noise-baseline control experiments**

Our goal is **not** to criticize prior work but to improve reliability in this emerging field.

---

## 📊 Performance Summary
Performance comparisons across inputs:

| Input Type | Expected Behavior | Observed Behavior |
|------------|------------------|------------------|
| EEG → text | High performance | Same as noise |
| Noise → text | Low performance | Same as EEG |
| Teacher-forcing | – | Inflated BLEU/ROUGE, unrealistic |

These findings demonstrate the need for **statistically grounded evaluation without teacher-forcing**.

---

## 🖼️ Figures
Model overview:  
![overview](https://github.com/NeuSpeech/EEG-To-Text/assets/151606332/57212488-b75f-44c7-a265-e2a51483e9f5)

Performance comparison:  
![performance](https://github.com/NeuSpeech/EEG-To-Text/assets/151606332/df58870c-5277-4935-8c66-15efd58e9283)

---

## 🔧 Usage
Please refer to the scripts in `main` for:
- Training with EEG or noise
- Non-teacher-forced inference
- Bootstrap-based statistical evaluation

We welcome community validation and extensions of our methodology.

---

## 🤝 Contact
For questions, issues, or collaboration:

📧 Hyejeong Jo — girlsending0@khu.ac.kr  
📧 Won Hee Lee — whlee@khu.ac.kr

---

## Acknowledgements
This work was supported in part by the Culture, Sports and Tourism Research and Development Program through Korea Creative Content Agency grant funded by the Ministry of Culture, Sports and Tourism under Grant RS-2023-00226263; and in part by the Institute of Information and Communications Technology Planning and Evaluation (IITP), funded by Korean Government (MSIT), through the Global AI Frontier Laboratory under Grant RS-2024-00509257, through the Information Technology Research Center (ITRC) under Grant IITP-2024-RS-2024-00438239, and through the AI Convergence Innovation Human Resources Development, Kyung Hee University under Grant RS-2022-00155911.

---

## 📝 Citation
If you use this repository, please cite:

```bibtex
@article{jo2025eegnoise,
  title={Evaluating EEG-to-text models through noise-based performance analysis},
  author={Jo, Hyejeong and Yang, Yiqian and Han, Juhyeok and Duan, Yiqun and Xiong, Hui and Lee, Won Hee},
  journal={Scientific Reports},
  year={2025},
  publisher={Nature Portfolio}
}
