import { useEffect, useRef, useState, useCallback } from "react";
import { io, Socket } from "socket.io-client";
import { Message } from "@/lib/api-client";

interface WebSocketMessage {
  type: "message" | "typing" | "session_ended" | "session_created";
  data: any;
}

export function useWebSocket(
  sessionId: string | null,
  onMessage?: (message: Message) => void,
) {
  const [isConnected, setIsConnected] = useState(false);
  const [typingUsers, setTypingUsers] = useState<Set<string>>(new Set());
  const wsRef = useRef<Socket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    if (!sessionId) return;

    const socketUrl =
      process.env.NEXT_PUBLIC_SOCKET_URL || "http://localhost:4000";
    const socket = io(socketUrl, {
      transports: ["websocket"],
      query: { sessionId },
    });
    wsRef.current = socket;

    socket.on("connect", () => {
      console.log("Socket connected");
      setIsConnected(true);
      socket.emit("join_session", {
        sessionToken: sessionId,
        role: "CUSTOMER",
      });
    });

    socket.on("new_message", (data: any) => {
      if (onMessage) {
        onMessage({
          id: data.id ?? `msg_${Date.now()}`,
          text: data.content ?? data.text ?? "",
          sender_type: data.sender?.role === "CUSTOMER" ? "CUSTOMER" : "STAFF",
          confidence: data.confidence ?? undefined,
          created_at: (
            data.sent_at ??
            data.created_at ??
            new Date().toISOString()
          ).toString(),
        });
      }
    });

    socket.on("typing", (data: any) => {
      setTypingUsers((prev) => {
        const newSet = new Set(prev);
        if (data?.is_typing) {
          newSet.add(data.user);
        } else {
          newSet.delete(data.user);
        }
        return newSet;
      });
    });

    socket.on("disconnect", () => {
      console.log("Socket disconnected");
      setIsConnected(false);
      reconnectTimeoutRef.current = setTimeout(() => connect(), 3000);
    });

    socket.on("connect_error", (error) => {
      console.error("Socket error:", error);
      setIsConnected(false);
    });
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
    if (wsRef.current?.connected) {
      wsRef.current.emit("typing", {
        is_typing: isTyping,
      });
    }
  }, []);

  const sendMessage = useCallback(
    (message: any) => {
      if (wsRef.current?.connected) {
        const content = message?.content ?? message?.text ?? "";
        if (!content) return;

        wsRef.current.emit("send_message", {
          sessionToken: sessionId,
          senderId:
            message?.senderId ??
            `customer_${sessionId ?? Date.now().toString()}`,
          role: message?.role ?? "CUSTOMER",
          content,
          modality: message?.modality ?? "TEXT",
          confidence: message?.confidence,
        });
      }
    },
    [sessionId],
  );

  return {
    isConnected,
    typingUsers,
    sendTyping,
    sendMessage,
  };
}
