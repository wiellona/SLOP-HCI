'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Mic, MicOff } from 'lucide-react';

interface VoiceInputProps {
  onTranscript: (text: string, confidence: number) => void;
  onError: (error: string) => void;
  disabled?: boolean;
  buttonSize?: number;
  sessionToken?: string;
}

export function VoiceInput({
  onTranscript,
  onError,
  disabled = false,
  buttonSize = 20,
  sessionToken,
}: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
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
        await sendAudioToServer(audioBlob);
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start(1000); // Collect data in 1-second chunks
      setIsRecording(true);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      onError('Tidak dapat mengakses mikrofon');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
    }
  };

    const sendAudioToServer = async (audioBlob: Blob) => {
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            if (sessionToken) {
            formData.append('session_token', sessionToken);
            }

            const response = await fetch('http://localhost:8000/v1/voice/transcribe', {
            method: 'POST',
            body: formData,
            });

            if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
            }

            const result = await response.json();
            console.log('Transcription result:', result);
            
            if (result.text && result.text.trim()) {
            onTranscript(result.text, result.confidence || 0.9);
            } else {
            onError('Tidak ada suara terdeteksi');
            }
        } catch (err) {
            console.error('Transcription error:', err);
            onError(err instanceof Error ? err.message : 'Gagal memproses suara');
        } finally {
            setIsProcessing(false);
        }
    };

  const toggleRecording = () => {
    if (disabled) return;
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <button
      onClick={toggleRecording}
      disabled={disabled || isProcessing}
      className="retro-btn"
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 46,
        flexShrink: 0,
        padding: 0,
        background: isRecording ? '#e05528' : undefined,
        color: isRecording ? '#fff8f0' : undefined,
      }}
    >
      {isProcessing ? (
        <div style={{ width: buttonSize, height: buttonSize, border: '2px solid #fff8f0', borderTop: '2px solid transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
      ) : isRecording ? (
        <MicOff size={buttonSize} />
      ) : (
        <Mic size={buttonSize} />
      )}
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </button>
  );
}