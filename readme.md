# AI-DRIVEN SUBTITLE GENERATION USING SELF-SUPERVISED LEARNING
---
Team Members:
1. Preeti Adhikari &emsp; [ACE078BCT050]
2. Rohan Basnet &emsp; [ACE078BCT053]
3. Sabal Gautam &emsp; [ACE078BCT054]
4. Samir Pokharel &emsp; [ACE078BCT058]
<br>
Project Supervisor:<br>
Er. Ramesh Sharma<br>
Department of Electronics and Computer Engineering

---

This project uses self-supervised framework inspired from wav2vec 2.0. The webapp uses the SSL model as a backbone to transcribe Devanagari scripts from Nepali Audio.

## Implementation

1. Run Docker and login.
2. Open Terminal and run:
```
docker pull incingren/nepali-asr:latest
docker run -p 5000:5000 incingren/nepali-asr:latest
```
3. Go to: `https://localhost:5000`
4. Drag and drop or browse to the source file to generate transcript.

---
