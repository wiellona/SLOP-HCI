import { useEffect, useRef, useState, useCallback } from 'react';
import { apiClient, Message } from '@/lib/api-client';

interface WebSocketMessage {
  type: 'message' | 'typing' | 'session_ended' | 'session_created';
  data: any;
}

export function useWebSocket(sessionId: string | null, onMessage?: (message: Message) => void) {
  const [isConnected, setIsConnected] = useState(false);
  const [typingUsers, setTypingUsers] = useState<Set<string>>(new Set());
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    if (!sessionId) return;

    const wsUrl = apiClient.getWebSocketUrl(sessionId);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      // Authenticate
      ws.send(JSON.stringify({
        type: 'auth',
        token: localStorage.getItem('staff_token'),
      }));
    };

    ws.onmessage = (event) => {
      try {
        const data: WebSocketMessage = JSON.parse(event.data);
        switch (data.type) {
          case 'message':
            if (onMessage) onMessage(data.data);
            break;
          case 'typing':
            setTypingUsers((prev) => {
              const newSet = new Set(prev);
              if (data.data.is_typing) {
                newSet.add(data.data.user);
              } else {
                newSet.delete(data.data.user);
              }
              return newSet;
            });
            break;
          case 'session_ended':
            console.log('Session ended by other party');
            break;
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      // Attempt to reconnect after 3 seconds
      reconnectTimeoutRef.current = setTimeout(() => connect(), 3000);
    };
  }, [sessionId, onMessage]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendTyping = useCallback((isTyping: boolean) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'typing',
        is_typing: isTyping,
      }));
    }
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'message',
        data: message,
      }));
    }
  }, []);

  return {
    isConnected,
    typingUsers,
    sendTyping,
    sendMessage,
  };
}