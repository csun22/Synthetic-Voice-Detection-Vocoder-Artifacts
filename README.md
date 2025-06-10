# 🧠 Synthetic-Voice-Detection-Vocoder-Artifacts

---

## 📁 LibriSeVoc Dataset

1. **We are the first to identify neural vocoders as a source of features to expose synthetic human voices.**  
   Here are the differences shown by the six vocoders compared to the original audio:

   ![image](https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts/assets/90001788/6c3381c4-af7e-4ce2-a446-b3c76bf52aee)

2. **We provide LibriSeVoC** as a dataset of self-vocoding samples created with six state-of-the-art vocoders to highlight and exploit the vocoder artifacts.  
   The composition of the dataset is shown in the following table:

   <img width="1000" alt="image" src="https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts/assets/90001788/c74fdb20-a5b7-4109-b833-821dd8dd6230">

   The source of our dataset ground truth comes from **LibriTTS**. Therefore, we follow the naming logic of LibriTTS.  
   For example:  
   `27_123349_000006_000000.wav` →  
   - `27` is the reader's ID  
   - `123349` is the ID of the chapter

---

## 🎯 Deepfake Detection

We propose a new approach to detecting synthetic human voices by:

- Exposing signal artifacts left by neural vocoders
- Modifying and improving the RawNet2 baseline by adding multi-loss

✅ This lowers the error rate from **6.10% to 4.54%** on the **ASVspoof Dataset**.

Here is the framework of the proposed synthesized voice detection method:

<img width="1000" alt="image" src="https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts/assets/90001788/c46df06b-6d62-4b0f-a9d2-f5ffc4e378b9">

---

## 📄 Paper & Dataset

- 📘 Paper:  
  **[AI-Synthesized Voice Detection Using Neural Vocoder Artifacts – CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Sun_AI-Synthesized_Voice_Detection_Using_Neural_Vocoder_Artifacts_CVPRW_2023_paper.html)**

- 📦 Dataset:  
  **[Download LibriSeVoc](https://ubuffalo-my.sharepoint.com/:u:/g/personal/csun22_buffalo_edu/Ed6tZWB6urtVxZMyB4pXn30BIgfRZKkwE8-1lrTFjQFLxg?e=iauwl5)**

---

## 🛠️ Usage

### 🏋️‍♀️ To train the model, run:

```bash
python main.py --data_path /your/path/to/LibriSeVoc/ --model_save_path /your/path/to/models/
```

### 🧪 To test with your sample, run:

```bash
python eval.py --input_path /your/path/to/sample.wav --model_path /your/path/to/your_model.pth
```

### 📥 Pretrained Model Weights

Download the trained model weights from the link below:
```bash
https://ubuffalo-my.sharepoint.com/:u:/g/personal/csun22_buffalo_edu/Edo9IdZWWM1cQ7a8ChIzOwMBpSeQThRZv4ntnksi49dRRg?e=WDsrRN
```

### 🌐 In-the-Wild Testing

You can test audio samples live on our lab's Deepfake O Meter platform:

```bash
https://zinc.cse.buffalo.edu/ubmdfl/deep-o-meter/landing_page
```

## 📄 License

This repository is licensed under the **MIT License**.  
You are free to use, modify, and distribute the code with proper attribution.

- 🔗 [MIT License](https://opensource.org/licenses/MIT)  
