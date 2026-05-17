import { useState, useCallback, useRef } from 'react';

interface VoiceRecorderOptions {
    onTranscription?: (text: string, confidence: number) => void;
    onError?: (error: string) => void;
}

    export function useVoiceRecorder({ onTranscription, onError }: VoiceRecorderOptions = {}) {
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);

    const startRecording = useCallback(async () => {
        try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
            await sendToSTT(audioBlob);
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start(100); // Collect data every 100ms
        setIsRecording(true);
        } catch (err) {
        console.error('Failed to start recording:', err);
        onError?.('Microphone access denied or unavailable');
        }
    }, [onError]);

    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
        setIsProcessing(true);
        }
    }, [isRecording]);

    const sendToSTT = async (audioBlob: Blob) => {
        try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        const response = await fetch('http://localhost:8000/v1/voice/transcribe', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`STT failed: ${response.statusText}`);
        }

        const result = await response.json();
        setIsProcessing(false);
        
        if (result.text && result.text.trim()) {
            onTranscription?.(result.text, result.confidence || 0.8);
        }
        } catch (err) {
        console.error('STT error:', err);
        setIsProcessing(false);
        onError?.('Failed to transcribe audio');
        }
    };

    return {
        isRecording,
        isProcessing,
        startRecording,
        stopRecording,
    };
}