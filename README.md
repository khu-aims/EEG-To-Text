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






The **main branch** contains the final code for our "Are EEG-to-Text Models Working?" paper. 

Accepted by [IJCAI workshop 2024](https://github.com/user-attachments/files/16624318/IJCAI_hyejeongjo_poster_Final.pdf)

If you have any questions, you can write them in the Issues section or email Hyejeong Jo at girlsending0@khu.ac.kr.

check our new paper with full detailed comparison of different models on this task at [https://arxiv.org/abs/2405.06459](https://arxiv.org/abs/2405.06459)

overview
![image](https://github.com/NeuSpeech/EEG-To-Text/assets/151606332/57212488-b75f-44c7-a265-e2a51483e9f5)

performance
![image](https://github.com/NeuSpeech/EEG-To-Text/assets/151606332/df58870c-5277-4935-8c66-15efd58e9283)



# Correction on [(AAAI 2022) Open Vocabulary EEG-To-Text Decoding and Zero-shot sentiment classification](https://arxiv.org/abs/2112.02690)
# results and code is updated on **master** branch
# results and code is updated on **master** branch
# results and code is updated on **master** branch
**First of all, we are not pointing at others, we do this correction due to no offense, but a kind reminder of being careful of the string generation process. 
We repsect Mr. Wang very much, and appreciate his great contribution in this area.**

After scrutilizing [the original code shared by Wang Zhenhailong](https://github.com/MikeWangWZHL/EEG-To-Text), we discovered that the eval method have an unintentional but very serious mistake in generating predicted strings, which is using teacher forcing implicitly. 

The code which reaches my concern is:


```python
seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)
logits = seq2seqLMoutput.logits # bs*seq_len*voc_sz
probs = logits[0].softmax(dim = 1)
values, predictions = probs.topk(1)
predictions = torch.squeeze(predictions)
predicted_string = tokenizer.decode(predictions) 
```

Therefore resulting in [predictions like below](https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/results/task1_task2_taskNRv2-BrainTranslator_skipstep1-all_generation_results-7_22.txt#L61):

```
target string: It isn't that Stealing Harvard is a horrible movie -- if only it were that grand a failure!
predicted string:  was't a the. is was a bad place, it it it were a.. movie.
################################################


target string: It just doesn't have much else... especially in a moral sense.
predicted string:  was so't work the to to and not the country sense.
################################################


target string: Those unfamiliar with Mormon traditions may find The Singles Ward occasionally bewildering.
predicted string:  who with the history may be themselves Mormoning''s amusingering.
################################################


target string: Viewed as a comedy, a romance, a fairy tale, or a drama, there's nothing remotely triumphant about this motion picture.
predicted string:  the from a whole, it film, and comedy tale, and a tragic, it is nothing quite romantic about it. picture.
################################################


target string: But the talented cast alone will keep you watching, as will the fight scenes.
predicted string:  the most and of cannot not the entertained. and they the music against.
################################################


target string: It's solid and affecting and exactly as thought-provoking as it should be.
predicted string:  was a, it, it what it.provoking as it is be.
################################################


target string: Thanks largely to Williams, all the interesting developments are processed in 60 minutes -- the rest is just an overexposed waste of film.
predicted string:  to to the, the of films and in in in a minutes. and longest is a a afteragerposure, of time time
################################################


target string: Cantet perfectly captures the hotel lobbies, two-lane highways, and roadside cafes that permeate Vincent's days
predicted string: urtor was describes the spirit'sies and the ofstory streets, and the parking of areate the's life.</s>'sgggggggg,,,,,,,,,,,,,,</s>,,,,,
################################################


target string: An important movie, a reminder of the power of film to move us and to make us examine our values.
predicted string: nie part in " classic of the importance of the, shape people, our make us think our lives,
################################################


target string: Too much of this well-acted but dangerously slow thriller feels like a preamble to a bigger, more complicated story, one that never materializes.
predicted string:  bad of a is-known film not over- is like a film-ble to a much, more dramatic story. which that is endsizes.
```

In addition, we noticed that some people are using it as code base which generates concerning results. We are not condemning these researchers, we just want to notice them and maybe we can do something together to resolve this problem. 

[BELT Bootstrapping Electroencephalography-to-Language Decoding and Zero-Shot SenTiment Classification by Natural Language Supervision](https://arxiv.org/pdf/2309.12056)

[Aligning Semantic in Brain and Language: A Curriculum Contrastive Method for Electroencephalography-to-Text Generation](https://ieeexplore.ieee.org/iel7/7333/4359219/10248031.pdf)

[UniCoRN: Unified Cognitive Signal ReconstructioN bridging cognitive signals and human language](https://arxiv.org/pdf/2307.05355)

[Semantic-aware Contrastive Learning for Electroencephalography-to-Text Generation with Curriculum Learning](https://arxiv.org/pdf/2301.09237)

[DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation](https://arxiv.org/pdf/2309.14030)

We have written a corrected version to use model.generate to evaluate the model, the result is not so good. 
Basicly, we changed the model_decoding.py and eval_decoding.py to add model.generate for its originally nn.Module class model, and used model.generate to predict strings.

**We are open to everyone to scrutinize on this corrected code and run the code. Then, we will show the final performance of this model in this repo and formalize a technical paper.**
# We really appreciate the great contribution made by Mr. Wang, however, we should prevent others from continuing this misunderstanding. 

# Acknowledgements
This work was supported in part by the Culture, Sports and Tourism Research and Development Program through Korea Creative Content Agency grant funded by the Ministry of Culture, Sports and Tourism under Grant RS-2023-00226263; and in part by the Institute of Information and Communications Technology Planning and Evaluation (IITP), funded by Korean Government (MSIT), through the Global AI Frontier Laboratory under Grant RS-2024-00509257, through the Information Technology Research Center (ITRC) under Grant IITP-2024-RS-2024-00438239, and through the AI Convergence Innovation Human Resources Development, Kyung Hee University under Grant RS-2022-00155911.
