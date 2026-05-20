'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import Webcam from 'react-webcam';

import { apiClient, Message as APIMessage } from '@/lib/api-client';
import { CafeHeader } from '@/components/cafe/CafeHeader';
import { StatusBadge } from '@/components/cafe/StatusBadge';
import { AccessibilityToolbar } from '@/components/cafe/AccessibilityToolbar';
import { CafeCameraFeed } from '@/components/camera/CameraFeed';
import { BoundingBoxOverlay } from '@/components/camera/BoundingBoxOverlay';
import { CafeChatHistory } from '@/components/chat/ChatHistory';
import { CafeTranslationPreview } from '@/components/chat/TranslationPreview';
import { CafeSuggestionChips } from '@/components/chat/SuggestionChip';
import { CafeActionButtons } from '@/components/chat/ActionButtons';
import { CafeRecommendationQuestions } from '@/components/chat/RecommendationQuestion';
import { EditPanel } from '@/components/chat/EditPanel';
import { useWebSocket } from '@/app/hooks/useWebSocket';


interface LocalMessage {
  id: string;
  text: string;
  sender: 'customer' | 'staff';
  timestamp: Date;
  confidence?: number;
}

interface BoundingBoxType {
  x: number;
  y: number;
  width: number;
  height: number;
  is_occluded: boolean;
  occlusion_score: number;
}

interface StaffInfo {
  id: string;
  name: string;
  email: string;
}

export default function CustomerPage() {
  // State
  const [showEditPanel, setShowEditPanel] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(true);
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);
  const [currentTranslation, setCurrentTranslation] = useState('');
  const [sentenceBuffer, setSentenceBuffer] = useState<string[]>([]);
  const [lastWordTime, setLastWordTime] = useState<number>(Date.now());
  const [confidence, setConfidence] = useState(0);
  const [alternatives, setAlternatives] = useState<string[]>([]);
  const [isTracking, setIsTracking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [boundingBox, setBoundingBox] = useState<BoundingBoxType | null>(null);
  const [occlusionDetected, setOcclusionDetected] = useState(false);
  const [conversation, setConversation] = useState<LocalMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStaff, setCurrentStaff] = useState<StaffInfo | null>(null);
  const [isOnline, setIsOnline] = useState(true);
  const [currentTime, setCurrentTime] = useState(new Date());

  // Refs
  const webcamRef = useRef<Webcam>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const SENTENCE_GAP_MS = 2000;
  const POLL_INTERVAL_MS = 3000;

  // Load active session
  const loadActiveSession = useCallback(async () => {
    try {
      const sessions = await apiClient.getAllSessions();
      const activeSession = sessions.find(s => s.status === 'ACTIVE');

      if (activeSession) {
        setSessionId(activeSession.id);
        const msgs = await apiClient.getSessionMessages(activeSession.id);
        setConversation(
          msgs.map((msg: APIMessage) => ({
            id: msg.id,
            text: msg.text,
            sender: msg.sender_type === 'CUSTOMER' ? 'customer' : 'staff',
            timestamp: new Date(msg.created_at),
            confidence: msg.confidence ?? undefined,
          }))
        );
      }
    } catch (err) {
      console.error('Failed to load session:', err);
    }
  }, []);

  // WebSocket connection
  const initWebSocket = useCallback(() => {
  const mlWsUrl = process.env.NEXT_PUBLIC_ML_WS_URL || 'ws://localhost:8000';
  const wsUrl = `${mlWsUrl}/v1/sign/stream`;
  console.log('Connecting to WebSocket:', wsUrl);
  
  wsRef.current = new WebSocket(wsUrl);

  wsRef.current.onopen = () => {
    console.log('WebSocket connected, sending session token...');
    // Kirim session token setelah connection terbuka
    const sessionToken = sessionId || 'customer_session_' + Date.now();
    wsRef.current?.send(JSON.stringify({ session_token: sessionToken }));
    setIsWebSocketConnected(true);
  };

  wsRef.current.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      console.log('WebSocket message received:', data);
      
      switch (data.event) {
        case 'connected':
          console.log('Connected to sign service:', data.message);
          break;
        case 'text_streamed':
          if (data.partial_text) {
            setCurrentTranslation(data.partial_text);
            setConfidence(data.confidence_score || 0);
            setIsTracking(data.hand_detected || false);
            if (data.bounding_box) {
              setBoundingBox(data.bounding_box);
              setOcclusionDetected(data.bounding_box.is_occluded || false);
            }
          }
          break;
        case 'inference_complete':
          handleInferenceComplete(data);
          break;
        case 'error':
          console.error('Inference error:', data.error);
          setIsProcessing(false);
          break;
      }
    } catch (err) {
      console.error('Failed to parse message:', err);
    }
  };

  wsRef.current.onerror = (error) => {
    console.error('WebSocket error:', error);
    setIsWebSocketConnected(false);
  };

  wsRef.current.onclose = () => {
      console.log('WebSocket closed, attempting to reconnect...');
      setIsWebSocketConnected(false);
      // Reconnect after 3 seconds
      setTimeout(() => {
        if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
          initWebSocket();
        }
      }, 3000);
    };
  }, [sessionId]);

  const handleInferenceComplete = useCallback((data: any) => {
    const word = data.raw_prediction_text;
    const now = Date.now();

    console.log(`Inference complete: "${word}" with confidence ${data.final_confidence_score}`);

    if (data.bounding_box) {
      setBoundingBox(data.bounding_box);
      setOcclusionDetected(data.bounding_box.is_occluded || false);
    }
    if (now - lastWordTime > SENTENCE_GAP_MS) {
      setSentenceBuffer([word]);
      setCurrentTranslation(word);
    } else {
      const newBuffer = [...sentenceBuffer, word];
      setSentenceBuffer(newBuffer);
      setCurrentTranslation(newBuffer.join(' '));
    }
    setLastWordTime(now);
    setConfidence(data.final_confidence_score || 0);
    setIsProcessing(false);

    if (data.final_confidence_score < 0.85 && data.alternatives && data.alternatives.length > 0) {
      setAlternatives(data.alternatives);
    } else {
      setAlternatives([]);
    }
  }, [lastWordTime, sentenceBuffer, SENTENCE_GAP_MS]);

  const { isConnected: wsConnected, sendMessage: wsSendMessage } = useWebSocket(
  sessionId,
  useCallback((newMessage: Message) => {
    // Handle incoming messages via WebSocket
    setConversation((prev) => [
      ...prev,
      {
        id: newMessage.id,
        text: newMessage.text,
        sender: newMessage.sender_type === 'CUSTOMER' ? 'customer' : 'staff',
        timestamp: new Date(newMessage.created_at),
        confidence: newMessage.confidence,
      },
    ]);
  }, [])
);

  // Modify handleSend to use WebSocket
  const handleSend = useCallback(async () => {
    if (!currentTranslation.trim() || !sessionId) return;

    const text = currentTranslation.trim();
    const tempMessage: LocalMessage = {
      id: `temp_${Date.now()}`,
      text,
      sender: 'customer',
      timestamp: new Date(),
      confidence,
    };
    setConversation(prev => [...prev, tempMessage]);
    setCurrentTranslation('');
    setSentenceBuffer([]);
    setAlternatives([]);

    try {
      await apiClient.sendMessage(sessionId, text, 'CUSTOMER', confidence);
      // Send via WebSocket for real-time delivery
      if (wsConnected) {
        wsSendMessage({
          text,
          sender_type: 'CUSTOMER',
          confidence,
        });
      }
    } catch (err) {
      console.error('Failed to send message:', err);
      setConversation(prev => prev.filter(msg => msg.id !== tempMessage.id));
    }
  }, [currentTranslation, confidence, sessionId, wsConnected, wsSendMessage]);
  

  // Frame capture
  const startFrameCapture = useCallback(() => {
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);

    frameIntervalRef.current = setInterval(() => {
      if (!webcamRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        return;
      }

      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        try {
          // Convert base64 to binary
          const base64Data = imageSrc.split(',')[1];
          const binaryString = atob(base64Data);
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
          }
          
          // Send binary data
          wsRef.current.send(bytes);
        } catch (err) {
          console.error('Failed to send frame:', err);
        }
      }
    }, 100);
  }, []);

  const handleEditComplete = useCallback((newText: string) => {
    setCurrentTranslation(newText);
    setSentenceBuffer(newText.split(' '));
    setShowEditPanel(false);
  }, []);

  const handleSuggestionClick = useCallback((suggestion: string) => {
    setCurrentTranslation(suggestion);
    setSentenceBuffer(suggestion.split(' '));
    setAlternatives([]);
  }, []);

  const handleRecommendationClick = useCallback((question: string) => {
    setCurrentTranslation(question);
    setSentenceBuffer(question.split(' '));
  }, []);

  const resetChat = useCallback(() => {
    setConversation([]);
    setCurrentTranslation('');
    setSentenceBuffer([]);
    setAlternatives([]);
  }, []);

  const toggleCamera = useCallback(() => {
    setIsCameraActive(prev => !prev);
  }, []);

  // Time and online status
  useEffect(() => {
    const interval = setInterval(() => setCurrentTime(new Date()), 1000);
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      clearInterval(interval);
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Effects
  useEffect(() => {
    loadActiveSession();
    pollIntervalRef.current = setInterval(() => {
      if (sessionId) {
        apiClient.getSessionMessages(sessionId).then((msgs) => {
          setConversation(
            msgs.map((msg: APIMessage) => ({
              id: msg.id,
              text: msg.text,
              sender: msg.sender_type === 'CUSTOMER' ? 'customer' : 'staff',
              timestamp: new Date(msg.created_at),
              confidence: msg.confidence ?? undefined,
            }))
          );
        }).catch(console.error);
      } else {
        loadActiveSession();
      }
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [loadActiveSession, sessionId]);

  useEffect(() => {
    initWebSocket();
    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [initWebSocket]);

  useEffect(() => {
    if (isCameraActive && isWebSocketConnected) {
      startFrameCapture();
    } else if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  }, [isCameraActive, isWebSocketConnected, startFrameCapture]);

  // UI helpers
  const getCameraBorderColor = () => {
    if (!isTracking) return '#E8DCC8';
    if (occlusionDetected) return '#D4A843';
    if (confidence >= 0.9) return '#6B8C42';
    if (confidence >= 0.7) return '#D4A843';
    return '#B85C4A';
  };

  const getCameraBgColor = () => {
    if (!isTracking) return '#FAF6F0';
    if (occlusionDetected) return '#FDF5E6';
    if (confidence >= 0.9) return '#F0F5E8';
    if (confidence >= 0.7) return '#FDF5E6';
    return '#FDF0ED';
  };

  const getCameraStatusText = () => {
    if (!isTracking) return 'Tidak ada tangan terdeteksi';
    if (occlusionDetected) return 'Tangan terdeteksi - keluar dari frame';
    if (confidence >= 0.9) return 'Gesture terdeteksi dengan jelas';
    if (confidence >= 0.7) return 'Gesture terdeteksi - keyakinan sedang';
    return 'Gesture tidak terdeteksi dengan jelas';
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f8eddb' }}>
      <CafeHeader
        title="SLOP"
        subtitle="Customer Terminal | Gesture-to-Text"
        status={isWebSocketConnected ? 'online' : 'connecting'}
        onReset={resetChat}
      />

      {/* Main Layout */}
      <div style={{ 
        display: 'grid', 
        height: 'calc(100vh - 73px)',
        gridTemplateColumns: '1.6fr 1fr', 
        gap: 10, 
        padding: 12,
        overflow: 'auto'
      }}>
        {/* Left Section */}
        <div style={{ display: 'flex', flexDirection: 'column', flexWrap: 'wrap', minHeight: 0, gap: 10 }}>
          <div style={{ flex: 'auto', minHeight: 200}}>
            <CafeCameraFeed
              webcamRef={webcamRef}
              isActive={isCameraActive}
              onToggle={toggleCamera}
              borderColor={getCameraBorderColor()}
              backgroundColor={getCameraBgColor()}
              statusText={getCameraStatusText()}
              confidence={confidence}
              isTracking={isTracking}
            >
              <BoundingBoxOverlay
                boundingBox={boundingBox}
                confidence={confidence}
                occlusionDetected={occlusionDetected}
              />
            </CafeCameraFeed>
          </div>

          <div style={{ flexShrink: 1, marginTop: 'auto' }}>
            <CafeTranslationPreview
              translation={currentTranslation}
              isProcessing={isProcessing}
              onEdit={() => setShowEditPanel(true)}
            />
          </div>

          <div style={{ flexShrink: 1 }}>
            <CafeSuggestionChips
              currentWord={currentTranslation}
              suggestions={alternatives}
              onSelect={handleSuggestionClick}
            />
          </div>

          <div style={{ flexShrink: 0, marginTop: 'auto' }}>
            <CafeActionButtons 
              onSend={handleSend} 
              disabled={!currentTranslation || isProcessing || !sessionId} 
            />
          </div>
        </div>

        {/* Right Section */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <CafeChatHistory
            messages={conversation}
            emptyMessage="Belum ada pesan"
            emptySubmessage="Mulai dengan melakukan gesture"
          />
          <CafeRecommendationQuestions onSelectQuestion={handleRecommendationClick} />
        </div>
      </div>

      <StatusBadge isOnline={isOnline} currentTime={currentTime} />
      <AccessibilityToolbar />

      {showEditPanel && (
        <EditPanel
          initialText={currentTranslation}
          onSave={handleEditComplete}
          onClose={() => setShowEditPanel(false)}
        />
      )}
    </div>
  );
}