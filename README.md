<div align="center">

![SLOP-Logo](https://hackmd.io/_uploads/HJZ6Xdakzg.png)
### Sign Language Output Provider


Bridging communication between Deaf communities and the public through AI-powered real-time sign language and speech translation.

[![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-4285F4)](https://mediapipe.dev/)
[![Whisper](https://img.shields.io/badge/Whisper-STT-412991)](https://github.com/openai/whisper)
[![IndoBERT](https://img.shields.io/badge/IndoBERT-NLP-FF6F00)](https://huggingface.co/indobenchmark)
[![Siformer](https://img.shields.io/badge/Siformer-SLR-6C63FF)](#)
</div>

---

## Overview

SLOP (Sign Language Output Provider) is an AI-powered accessibility platform designed to facilitate two-way communication between Deaf or speech-impaired individuals and the general public, particularly in café and environments.

The system provides:

- **Sign-to-Text Translation**
- **Speech-to-Text Translation**
- **Real-Time Conversation Interface**
- **Human-in-the-Loop Validation**
- **Confidence-Based AI Feedback**
- **Accessibility-First Design**

SLOP aims to reduce communication barriers by translating Indonesian Sign Language (BISINDO) into text while simultaneously converting spoken language into readable text.

---

## Key Features

### Sign Language Recognition (SLR)

- Real-time hand and body landmark extraction using MediaPipe
- Skeleton-based gesture recognition
- BISINDO vocabulary support
- Confidence score visualization

### Speech-to-Text (STT)

- Indonesian speech recognition
- Noise-tolerant transcription
- Push-to-talk interaction mode

### Dual-Screen Communication

The system separates communication into two perspectives:

#### Customer Screen

For Deaf or speech-impaired users:

- Camera-based sign language input
- Live gesture feedback
- Editable AI-generated text
- Manual fallback input

#### Staff Screen

For café staff:

- Voice input
- Real-time transcription
- Conversation history
- Accessibility-friendly interaction

### Two-Stage Validation

Messages are never sent automatically.

Users can:

1. Review AI-generated text
2. Edit incorrect predictions
3. Explicitly press **Send**

This prevents AI misinterpretations from being transmitted without user confirmation.

---

## System Architecture

```text
┌─────────────────┐
│ Customer Screen │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    MediaPipe    │
│ Landmark Extract│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Siformer     │
│ Sign Recognition│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ IndoBERT / LLM  │
│ Text Refinement │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Conversation UI │
└─────────────────┘


┌───────────────┐
│ Staff Screen  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Whisper STT │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Conversation  │
│      UI       │
└───────────────┘
```

---

## AI Technologies

### Sign Language Pipeline

| Component | Technology |
|-----------|------------|
| Landmark Extraction | MediaPipe |
| Sign Recognition | Siformer |
| Feature Type | Skeleton-Based Coordinates |
| Input | Hand, Pose, Face Landmarks |

### Speech Pipeline

| Component | Technology |
|-----------|------------|
| Speech Recognition | OpenAI Whisper |
| Language | Indonesian |
| Input | Microphone Audio |

### Language Processing

| Component | Technology |
|-----------|------------|
| Text Refinement | IndoBERT / Local LLM |
| Output | Natural Indonesian Sentences |

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+
- npm or yarn
- Webcam
- Microphone

---

### Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend will be available at:

```text
http://localhost:3000
```

---

### Backend Setup

Create virtual environment:

```bash
python -m venv venv
```

Activate:

```bash
# Windows
venv\Scripts\activate

# Linux / MacOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server:

```bash
uvicorn main:app --reload
```

Backend will be available at:

```text
http://localhost:8000
```

---

## Accessibility Features

### Confidence-Based Feedback

| Confidence | Status |
|------------|--------|
| > 90% | 🟢 High Confidence |
| 70–89% | 🟡 Medium Confidence |
| < 70% | 🔴 Low Confidence |

### Manual Fallback

When AI cannot confidently recognize input, users can:

- Edit generated text
- Type manually
- Continue communication without AI assistance

### Error Prevention

- Explicit Send Confirmation
- Editable Predictions
- Visual Detection Feedback
- Real-Time Status Indicators

---

## Contributors

| Name | Role |
|--------|--------|
| Arsinta Kirana Nisa | Backend |
| Azra Nabila Azzahra | Frontend & Integrating |
| Siti Amalia Nurfaidah | Speech-to-Text |
| Wiellona Darlene Oderia Saragih | Sign-to-Text Siformer |

---

## License

This project was developed as part of the Human-Computer Interaction course at Universitas Indonesia.

For academic and research purposes only.

---

<div align="center">

### Breaking Communication Barriers Through AI

SLOP — Sign Language Output Provider

</div>
