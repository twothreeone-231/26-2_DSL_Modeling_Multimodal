---
language:
- en
tags:
- audio
- audio-classification
- antispoofing
- deepfake-detection
- speech
license: other
pipeline_tag: audio-classification
---

# DF Arena 1B - Antispoofing Model

We are excited to release DF Arena 1B Universal Antispoofing model 🔥trained on traditional speech antispoofing datasets in addition to singing and environmental deepfake data. 
Check out the release on [DF Arena leaderboard](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena) 

# Training Data

- **ASVspoof 2019, 2024**
- **Codecfake**
- **LibriSeVoc**
- **DFADD**
- **CTRSVDD**
- **SpoofCeleb**
- **MLAAD**
- **EnvSDD**

## Usage
```python
from transformers import pipeline
import librosa

 #load model
pipe = pipeline("antispoofing", model="Speech-Arena-2025/DF_Arena_1B_V_1", trust_remote_code=True, device='cuda')
audio, sr = librosa.load("sample.wav", sr=16000)
result = pipe(audio)
print(result)
# Output: 
{'label': 'spoof', 'logits': [[1.5515458583831787, -1.2254822254180908]], 'score': 0.9414217472076416, 'all_scores': {'spoof': 0.9414217472076416, 'bonafide': 0.05857823044061661}}
```

# Evaluation

| Dataset                | EER (%) | F1-score | Accuracy (%) |
|-------------------------|----------|-----------|---------------|
| dfadd               | 0.00     | 0.9993    | 99.97         |
| add_2023_round_2    | 11.54    | 0.9188    | 88.46         |
| codecfake           | 8.37     | 0.8695    | 91.63         |
| asvspoof_2021_la    | 4.66     | 0.8037    | 95.34         |
| in_the_wild         | 0.91     | 0.9928    | 99.10         |
| asvspoof_2019       | 1.14     | 0.9473    | 98.86         |
| add_2022_track_1    | 22.21    | 0.6678    | 77.79         |
| fake_or_real        | 2.92     | 0.9711    | 97.11         |
| asvspoof_2024       | 17.25    | 0.6615    | 82.75         |
| add_2022_track_3    | 2.20     | 0.9357    | 97.80         |
| add_2023_round_1    | 5.08     | 0.9639    | 94.92         |
| librisevoc          | 0.15     | 0.9958    | 99.84         |
| asvspoof_2021_df    | 1.75     | 0.7577    | 98.25         |
| sonar               | 1.09     | 0.9903    | 98.89         |
| Average               | 5.919     | 0.8863    | 94.079         |
| Pooled              | 9.52     | 0.81    | 90.47         |










## License

We use a non-commercial license which can be found [here](./LICENSE.txt)

## Contact

For questions or issues, please open an issue on the model repository or contact us at ajinkya.kulkarni@idiap.ch.

Stay tuned for upcoming versions of our models!

## Citation

If you use this model in your work, it can be cited as :

```bibtex
@misc{kulkarni2026compactsslbackbonesmatter,
      title={Do Compact SSL Backbones Matter for Audio Deepfake Detection? A Controlled Study with RAPTOR}, 
      author={Ajinkya Kulkarni and Sandipana Dowerah and Atharva Kulkarni and Tanel Alumäe and Mathew Magimai Doss},
      year={2026},
      eprint={2603.06164},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.06164}, 
}
```