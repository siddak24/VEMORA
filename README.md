# VEMORA

VEMORA is a multimodal AI smart-glasses project being developed as a final-year project.

The long-term goal is to build an intelligent wearable assistant capable of:

- Voice interaction
- Computer vision
- Persistent personal memory
- Context-aware assistance
- Gesture-based interaction
- Local AI/SLM inference
- Optional cloud AI fallback
- Edge deployment on standalone hardware

## Current Prototype

The current laptop prototype simulates the smart glasses using:

- Laptop microphone
- Laptop webcam
- Laptop speakers
- Local speech-to-text
- Gemini API
- Local text-to-speech

### Current pipeline

```text
Microphone
    ↓
faster-whisper
    ↓
Speech text
    ↓
Gemini API
    ↓
Response
    ↓
Local TTS
    ↓
Speaker
