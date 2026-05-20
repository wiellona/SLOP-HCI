import axios from 'axios';

const ML_SERVICE_URL = process.env.NEXT_PUBLIC_ML_SERVICE_URL || 'http://localhost:8000';

export interface VoiceRecognitionResult {
    inference_id: string;
    session_token: string;
    raw_prediction_text: string;
    confidence_score: number;
    model_version: string;
    inference_latency_ms: number;
}

class VoiceService {
    private mediaRecorder: MediaRecorder | null = null;
    private audioChunks: Blob[] = [];
    private isRecording = false;
    private sessionToken: string | null = null;
    private onResultCallback: ((text: string, confidence: number) => void) | null = null;
    private onErrorCallback: ((error: string) => void) | null = null;

    setSessionToken(token: string) {
        this.sessionToken = token;
    }

    async startRecording(
        onResult: (text: string, confidence: number) => void,
        onError?: (error: string) => void
    ) {
        this.onResultCallback = onResult;
        this.onErrorCallback = onError || null;
        this.audioChunks = [];

        try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.mediaRecorder = new MediaRecorder(stream);
        
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
            this.audioChunks.push(event.data);
            }
        };

        this.mediaRecorder.onstop = async () => {
            await this.processAudio();
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };

        this.mediaRecorder.start(1000); // Record in 1-second chunks
        this.isRecording = true;
        
        } catch (error) {
        console.error('Error accessing microphone:', error);
        if (this.onErrorCallback) {
            this.onErrorCallback('Tidak dapat mengakses mikrofon');
        }
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
        this.mediaRecorder.stop();
        this.isRecording = false;
        }
    }

    private async processAudio() {
        if (this.audioChunks.length === 0) return;

        // Combine audio chunks into single WAV
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        const wavBlob = await this.convertToWav(audioBlob);
        
        // Convert to base64
        const reader = new FileReader();
        reader.readAsDataURL(wavBlob);
        reader.onloadend = async () => {
        const base64Audio = reader.result as string;
        
        try {
            const response = await axios.post<VoiceRecognitionResult>(
            `${ML_SERVICE_URL}/v1/voice/transcribe`,
            {
                audio_base64: base64Audio,
                session_token: this.sessionToken,
            },
            {
                headers: { 'Content-Type': 'application/json' }
            }
            );
            
            if (response.data.raw_prediction_text && this.onResultCallback) {
            this.onResultCallback(
                response.data.raw_prediction_text,
                response.data.confidence_score
            );
            }
        } catch (error) {
            console.error('Voice recognition error:', error);
            if (this.onErrorCallback) {
            this.onErrorCallback('Gagal memproses suara');
            }
        }
        };
    }

    private async convertToWav(blob: Blob): Promise<Blob> {
        // Simple conversion: create WAV header + PCM data
        const audioContext = new AudioContext({ sampleRate: 16000 });
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Resample to 16kHz mono
        const offlineContext = new OfflineAudioContext(
        1, // mono
        audioBuffer.duration * 16000,
        16000
        );
        const source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start();
        
        const renderedBuffer = await offlineContext.startRendering();
        const pcmData = renderedBuffer.getChannelData(0);
        
        // Convert float32 to int16
        const int16Data = new Int16Array(pcmData.length);
        for (let i = 0; i < pcmData.length; i++) {
        int16Data[i] = Math.max(-32768, Math.min(32767, Math.floor(pcmData[i] * 32768)));
        }
        
        // Create WAV file
        const wavBlob = this.createWavBlob(int16Data, 16000);
        return wavBlob;
    }

    private createWavBlob(samples: Int16Array, sampleRate: number): Blob {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        
        // RIFF chunk
        this.writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        this.writeString(view, 8, 'WAVE');
        
        // fmt sub-chunk
        this.writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        
        // data sub-chunk
        this.writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);
        
        // Write samples
        for (let i = 0; i < samples.length; i++) {
        view.setInt16(44 + i * 2, samples[i], true);
        }
        
        return new Blob([buffer], { type: 'audio/wav' });
    }

    private writeString(view: DataView, offset: number, str: string) {
        for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
        }
    }
}

export const voiceService = new VoiceService();