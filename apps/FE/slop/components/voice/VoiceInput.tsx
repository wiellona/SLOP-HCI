'use client';

import { useState, useCallback, useEffect } from 'react';
import { Mic, MicOff, Loader2 } from 'lucide-react';

interface VoiceInputProps {
  onTranscript: (text: string, confidence: number) => void;
  onError?: (error: string) => void;
  disabled?: boolean;
  buttonSize?: number;
}

export function VoiceInput({ 
  onTranscript, 
  onError, 
  disabled = false,
  buttonSize = 24 
}: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setAudioChunks(prev => [...prev, event.data]);
        }
      };
      
      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        await sendToSTT(audioBlob);
        
        // Clean up
        stream.getTracks().forEach(track => track.stop());
        setAudioChunks([]);
      };
      
      recorder.start(100); // Collect data in 100ms chunks
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (err) {
      console.error('Failed to start recording:', err);
      onError?.('Microphone access denied. Please check permissions.');
    }
  }, [audioChunks, onError]);

  const stopRecording = useCallback(() => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
      setIsProcessing(true);
      setMediaRecorder(null);
    }
  }, [mediaRecorder, isRecording]);

  const sendToSTT = async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    try {
      const response = await fetch('http://localhost:8000/v1/voice/transcribe', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`STT failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (result.text && result.text.trim()) {
        onTranscript(result.text, result.confidence);
      } else {
        onError?.('No speech detected. Please try again.');
      }
    } catch (err) {
      console.error('STT error:', err);
      onError?.('Failed to transcribe audio. Please check if ML service is running.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
      }
    };
  }, [mediaRecorder, isRecording]);

  const handleClick = () => {
    if (disabled || isProcessing) return;
    
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <button
      onClick={handleClick}
      disabled={disabled || isProcessing}
      style={{
        width: 46,
        flexShrink: 0,
        background: isRecording ? '#b85c4a' : (isProcessing ? '#a08060' : '#efb36d'),
        border: '2px solid #2b1d1d',
        boxShadow: isRecording ? 'none' : '3px 3px 0 #2b1d1d',
        cursor: (disabled || isProcessing) ? 'not-allowed' : 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transform: isRecording ? 'translate(3px,3px)' : 'none',
        transition: 'all 0.1s',
      }}
    >
      {isProcessing ? (
        <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
      ) : isRecording ? (
        <MicOff size={buttonSize} color="#fff8f0" />
      ) : (
        <Mic size={buttonSize} color="#2b1d1d" />
      )}
    </button>
  );
}

// Add spinning animation CSS
const style = document.createElement('style');
style.textContent = `
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
`;
document.head.appendChild(style);