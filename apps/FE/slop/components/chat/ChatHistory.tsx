'use client';

import { useRef, useEffect } from 'react';
import { MessageSquare } from 'lucide-react';

interface Message {
  id: string;
  text: string;
  sender: 'customer' | 'staff';
  timestamp: Date;
  confidence?: number;
}

interface CafeChatHistoryProps {
  messages: Message[];
  emptyMessage?: string;
  emptySubmessage?: string;
}

export function CafeChatHistory({
  messages,
  emptyMessage = 'Belum ada pesan',
  emptySubmessage = 'Mulai dengan melakukan gesture',
}: CafeChatHistoryProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="retro-window" style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
      <div className="retro-titlebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="retro-titlebar-icon" />
          <span className="retro-titlebar-title">chat_history.log</span>
        </div>
        <div className="retro-winctrls">
          <button className="retro-winctrl">-</button>
          <button className="retro-winctrl">+</button>
        </div>
      </div>

      {messages.length === 0 ? (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          flex: 1, padding: 24, gap: 10,
        }}>
          <MessageSquare size={32} color="#c4a87d" />
          <p style={{ fontFamily: 'VT323, monospace', fontSize: 20, color: '#a08060', margin: 0 }}>
            {emptyMessage.toUpperCase()}
          </p>
          <p style={{ fontFamily: 'VT323, monospace', fontSize: 16, color: '#c4a87d', margin: 0 }}>
            {emptySubmessage}
          </p>
        </div>
      ) : (
        <div
          ref={containerRef}
          style={{ flex: 1, overflowY: 'auto', padding: 10, display: 'flex', flexDirection: 'column', gap: 10 }}
        >
          {messages.map((msg) => {
            const isCustomer = msg.sender === 'customer';
            return (
              <div key={msg.id} style={{ display: 'flex', justifyContent: isCustomer ? 'flex-end' : 'flex-start' }}>
                <div className={isCustomer ? 'retro-chat-customer' : 'retro-chat-staff'}>
                  <p style={{ margin: 0, fontSize: 18, lineHeight: 1.3 }}>{msg.text}</p>
                  {msg.confidence && isCustomer && (
                    <div style={{ marginTop: 4, fontSize: 13, opacity: 0.7 }}>
                      CONF: {Math.round(msg.confidence * 100)}%
                    </div>
                  )}
                  <div style={{ marginTop: 4, fontSize: 13, opacity: 0.55, textAlign: isCustomer ? 'right' : 'left' }}>
                    [{msg.timestamp.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })}]
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
