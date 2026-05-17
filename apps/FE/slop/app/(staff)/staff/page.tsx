'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import {
  Mic,
  Send,
  LogOut,
  User,
  PlusCircle,
  ChevronDown,
  MicOff,
} from 'lucide-react';
import { apiClient, Conversation, Message } from '@/lib/api-client';
import { CafeChatHistory } from '@/components/chat/ChatHistory';
import { useWebSocket } from '@/app/hooks/useWebSocket';
import axios from 'axios';

interface LocalMessage {
  id: string;
  text: string;
  sender: 'customer' | 'staff';
  timestamp: Date;
  confidence?: number;
}

// Login Form component
function LoginForm({ onLogin }: { onLogin: () => Promise<void> }) {
  const [isRegister, setIsRegister] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [loginId, setLoginId] = useState('');
  const [loginPassword, setLoginPassword] = useState('');

  const [registerId, setRegisterId] = useState('');
  const [registerEmail, setRegisterEmail] = useState('');
  const [registerName, setRegisterName] = useState('');
  const [registerPassword, setRegisterPassword] = useState('');
  const [registerConfirmPassword, setRegisterConfirmPassword] = useState('');

  const generateSuggestedId = () => {
    const random = Math.random().toString(36).substring(2, 6).toUpperCase();
    setRegisterId(`STAFF_${Date.now()}_${random}`);
  };

  useEffect(() => {
    if (isRegister) generateSuggestedId();
  }, [isRegister]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      await apiClient.login({ id: loginId, password: loginPassword });
      await onLogin();
    } catch (err: any) {
      setError(err.message || 'Login gagal');
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    if (registerPassword !== registerConfirmPassword) {
      setError('Password tidak cocok');
      setLoading(false);
      return;
    }
    try {
      await apiClient.register({
        id: registerId,
        email: registerEmail,
        password: registerPassword,
        display_name: registerName,
        role: 'STAFF',
      });
      await onLogin();
    } catch (err: any) {
      setError(err.message || 'Registrasi gagal');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#d8a8b8', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}>
      <div className="retro-window" style={{ width: '100%', maxWidth: 420 }}>
        <div className="retro-titlebar">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="retro-titlebar-icon" />
            <span className="retro-titlebar-title">
              {isRegister ? 'register_staff.exe' : 'login.exe'}
            </span>
          </div>
          <div className="retro-winctrls">
            <button className="retro-winctrl">-</button>
            <button className="retro-winctrl">+</button>
            <button className="retro-winctrl" style={{ background: '#e05528', color: '#fff8f0' }}>x</button>
          </div>
        </div>

        <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ textAlign: 'center', marginBottom: 4 }}>
            <p style={{ fontFamily: 'VT323, monospace', fontSize: 14, color: '#a08060', margin: 0, textTransform: 'uppercase', letterSpacing: 1 }}>
              SLOP Staff Terminal v1.0
            </p>
          </div>

          {isRegister ? (
            <form onSubmit={handleRegister} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div>
                <label className="retro-label" style={{ display: 'block', marginBottom: 4 }}>ID Staff</label>
                <div style={{ display: 'flex', gap: 6 }}>
                  <input
                    type="text"
                    value={registerId}
                    onChange={(e) => setRegisterId(e.target.value)}
                    className="retro-input"
                    style={{ flex: 1 }}
                    required
                  />
                  <button type="button" onClick={generateSuggestedId} className="retro-btn" style={{ whiteSpace: 'nowrap', fontSize: 15 }}>
                    GEN
                  </button>
                </div>
              </div>
              {[
                { label: 'Email', value: registerEmail, set: setRegisterEmail, type: 'email' },
                { label: 'Nama Lengkap', value: registerName, set: setRegisterName, type: 'text' },
                { label: 'Password', value: registerPassword, set: setRegisterPassword, type: 'password' },
                { label: 'Konfirmasi Password', value: registerConfirmPassword, set: setRegisterConfirmPassword, type: 'password' },
              ].map(({ label, value, set, type }) => (
                <div key={label}>
                  <label className="retro-label" style={{ display: 'block', marginBottom: 4 }}>{label}</label>
                  <input
                    type={type}
                    value={value}
                    onChange={(e) => set(e.target.value)}
                    className="retro-input"
                    required
                  />
                </div>
              ))}
              {error && (
                <div className="retro-inset" style={{ background: '#fdf0ed', borderColor: '#b85c4a' }}>
                  <p style={{ fontFamily: 'VT323, monospace', fontSize: 16, color: '#b85c4a', margin: 0 }}>
                    ERROR: {error.toUpperCase()}
                  </p>
                </div>
              )}
              <button type="submit" disabled={loading} className="retro-btn-primary" style={{ width: '100%', marginTop: 4 }}>
                {loading ? 'LOADING...' : 'REGISTER'}
              </button>
              <button type="button" onClick={() => setIsRegister(false)} className="retro-btn" style={{ width: '100%' }}>
                SUDAH PUNYA AKUN? LOGIN
              </button>
            </form>
          ) : (
            <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div>
                <label className="retro-label" style={{ display: 'block', marginBottom: 4 }}>ID Staff</label>
                <input
                  type="text"
                  value={loginId}
                  onChange={(e) => setLoginId(e.target.value)}
                  className="retro-input"
                  placeholder="STAFF_xxx_xxx"
                  required
                />
              </div>
              <div>
                <label className="retro-label" style={{ display: 'block', marginBottom: 4 }}>Password</label>
                <input
                  type="password"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  className="retro-input"
                  required
                />
              </div>
              {error && (
                <div className="retro-inset" style={{ background: '#fdf0ed', borderColor: '#b85c4a' }}>
                  <p style={{ fontFamily: 'VT323, monospace', fontSize: 16, color: '#b85c4a', margin: 0 }}>
                    ERROR: {error.toUpperCase()}
                  </p>
                </div>
              )}
              <button type="submit" disabled={loading} className="retro-btn-primary" style={{ width: '100%', marginTop: 4 }}>
                {loading ? 'LOADING...' : 'LOGIN'}
              </button>
              <button type="button" onClick={() => setIsRegister(true)} className="retro-btn" style={{ width: '100%' }}>
                BELUM PUNYA AKUN? REGISTER
              </button>
            </form>
          )}
        </div>

        <div className="retro-statusbar">
          <span>SLOP v1.0</span>
          <span className="retro-statusbar-cell">Staff Auth System</span>
        </div>
      </div>
    </div>
  );
}

// Session Selector component
function SessionSelector({
  sessions,
  currentSession,
  onSelectSession,
  onCreateSession,
  loading,
}: {
  sessions: Conversation[];
  currentSession: Conversation | null;
  onSelectSession: (session: Conversation) => void;
  onCreateSession: () => void;
  loading: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setIsOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <div style={{ position: 'relative' }} ref={ref}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="retro-btn"
        style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 16 }}
      >
        <User size={13} />
        {currentSession
          ? `SESSION: ...${currentSession.session_token.slice(-6)}`
          : 'PILIH SESSION'}
        <ChevronDown size={13} style={{ transform: isOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.1s' }} />
      </button>

      {isOpen && (
        <div
          className="retro-window"
          style={{ position: 'absolute', right: 0, top: 'calc(100% + 6px)', width: 300, zIndex: 50 }}
        >
          <div className="retro-titlebar" style={{ fontSize: 15 }}>
            <span>sessions.dat</span>
          </div>

          <div style={{ padding: 6, borderBottom: '2px solid #2b1d1d' }}>
            <button
              onClick={() => { onCreateSession(); setIsOpen(false); }}
              disabled={loading}
              className="retro-btn"
              style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center', fontSize: 16 }}
            >
              <PlusCircle size={13} />
              {loading ? 'LOADING...' : 'BUAT SESSION BARU'}
            </button>
          </div>

          <div style={{ maxHeight: 220, overflowY: 'auto' }}>
            {sessions.length === 0 ? (
              <p style={{ padding: '10px 12px', fontFamily: 'VT323, monospace', fontSize: 17, color: '#a08060', margin: 0 }}>
                Belum ada session
              </p>
            ) : (
              sessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => { onSelectSession(session); setIsOpen(false); }}
                  style={{
                    display: 'flex',
                    width: '100%',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '7px 10px',
                    fontFamily: 'VT323, monospace',
                    fontSize: 17,
                    background: currentSession?.id === session.id ? '#efb36d' : '#f4dcc8',
                    border: 'none',
                    borderBottom: '1px solid #2b1d1d',
                    cursor: 'pointer',
                    color: '#2b1d1d',
                    textAlign: 'left',
                  }}
                >
                  <span>...{session.session_token.slice(-10)}</span>
                  <span style={{
                    fontFamily: 'VT323, monospace',
                    fontSize: 14,
                    color: session.status === 'ACTIVE' ? '#6B8C42' : '#a08060',
                  }}>
                    {session.status === 'ACTIVE' ? '[ACTIVE]' : '[ENDED]'}
                  </span>
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Main Staff Page
export default function StaffPage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState<{ id: string; display_name: string; role: string } | null>(null);
  const [sessions, setSessions] = useState<Conversation[]>([]);
  const [currentSession, setCurrentSession] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isVoiceMode, setIsVoiceMode] = useState(true);
  const [loading, setLoading] = useState(false);

  const loadMessages = useCallback(async (conversationId: string) => {
    try {
      const data = await apiClient.getSessionMessages(conversationId);
      setMessages(
        data.map((msg: Message) => ({
          id: msg.id,
          text: msg.text,
          sender: msg.sender_type === 'CUSTOMER' ? 'customer' : 'staff',
          timestamp: new Date(msg.created_at),
          confidence: msg.confidence ?? undefined,
        }))
      );
    } catch (err) {
      console.error('Failed to load messages:', err);
    }
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      const data = await apiClient.getAllSessions();
      setSessions(data);
      const activeSession = data.find((s: Conversation) => s.status === 'ACTIVE');
      if (activeSession && !currentSession) {
        setCurrentSession(activeSession);
        loadMessages(activeSession.id);
      }
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  }, [currentSession, loadMessages]);

  const handleLogin = useCallback(async () => {
    const user = apiClient.getCurrentUser();
    setCurrentUser(user);
    setIsAuthenticated(true);
    await loadSessions();
  }, [loadSessions]);

  const handleLogout = useCallback(() => {
    apiClient.logout();
    setIsAuthenticated(false);
    setCurrentUser(null);
    setSessions([]);
    setCurrentSession(null);
    setMessages([]);
  }, []);

  const handleCreateSession = useCallback(async () => {
    setLoading(true);
    try {
      const newSession = await apiClient.createSession();
      console.log('New session created:', newSession);
      
      // Refresh sessions list
      await loadSessions();
      
      // Get the updated sessions list and find the new session
      const updatedSessions = await apiClient.getAllSessions();
      const session = updatedSessions.find((s: Conversation) => s.id === newSession.session_id);
      
      if (session) {
        setCurrentSession(session);
        setMessages([]);
        console.log('Session selected:', session);
      }
    } catch (err) {
      console.error('Failed to create session:', err);
      alert('Gagal membuat session baru');
    } finally {
      setLoading(false);
    }
  }, [loadSessions]);

  const handleSelectSession = useCallback(async (session: Conversation) => {
    setCurrentSession(session);
    await loadMessages(session.id);
  }, [loadMessages]);

  const handleSend = useCallback(async () => {
    if (!currentMessage.trim() || !currentSession) return;
    const text = currentMessage.trim();
    setCurrentMessage('');
    
    const tempMessage: LocalMessage = {
      id: `temp_${Date.now()}`,
      text,
      sender: 'staff',
      timestamp: new Date()
    };
    setMessages((prev) => [...prev, tempMessage]);
    
    try {
      await apiClient.sendMessage(currentSession.id, text, 'STAFF');
      await loadMessages(currentSession.id);
    } catch (err) {
      console.error('Failed to send message:', err);
      setMessages((prev) => prev.filter((msg) => msg.id !== tempMessage.id));
    }
  }, [currentMessage, currentSession, loadMessages]);

  const handleEndSession = useCallback(async () => {
    if (!currentSession) return;
    try {
      await apiClient.endSession(currentSession.id);
      await loadSessions();
      setCurrentSession(null);
      setMessages([]);
    } catch (err) {
      console.error('Failed to end session:', err);
    }
  }, [currentSession, loadSessions]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const toggleRecording = useCallback(() => {
    setIsRecording((prev) => !prev);
    if (!isRecording) {
      setTimeout(() => {
        setCurrentMessage('Baik, pesanan Anda sedang diproses. Ada yang lain?');
        setIsRecording(false);
      }, 2000);
    }
  }, [isRecording]);

  // Poll for new messages
  useEffect(() => {
    if (!currentSession) return;
    
    const interval = setInterval(() => {
      loadMessages(currentSession.id);
    }, 3000);
    
    return () => clearInterval(interval);
  }, [currentSession, loadMessages]);

  if (!isAuthenticated) return <LoginForm onLogin={handleLogin} />;

  const isActive = currentSession?.status === 'ACTIVE';

  return (
    <div style={{ minHeight: '100vh', background: '#d8a8b8' }}>
      {/* Header */}
      <div className="retro-window" style={{ borderRadius: 0, borderLeft: 'none', borderRight: 'none', borderTop: 'none', boxShadow: '0 4px 0px #2b1d1d' }}>
        <div className="retro-titlebar">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="retro-titlebar-icon" />
            <span className="retro-titlebar-title">slop_staff_terminal.exe</span>
          </div>
          <div className="retro-winctrls">
            <button className="retro-winctrl">-</button>
            <button className="retro-winctrl">+</button>
            <button className="retro-winctrl" style={{ background: '#e05528', color: '#fff8f0' }}>x</button>
          </div>
        </div>

        <div className="retro-menubar">
          <span className="retro-menuitem">File</span>
          <span className="retro-menuitem">Session</span>
          <span className="retro-menuitem">Help</span>

          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontFamily: 'VT323, monospace', fontSize: 16, color: '#a08060' }}>
              USER: {(currentUser?.display_name || currentUser?.id || '').toUpperCase()}
            </span>

            <SessionSelector
              sessions={sessions}
              currentSession={currentSession}
              onSelectSession={handleSelectSession}
              onCreateSession={handleCreateSession}
              loading={loading}
            />

            {isActive && (
              <button onClick={handleEndSession} className="retro-btn" style={{ fontSize: 15, color: '#b85c4a', borderColor: '#b85c4a' }}>
                AKHIRI SESSION
              </button>
            )}

            <button onClick={handleLogout} className="retro-btn" style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 15 }}>
              <LogOut size={12} />
              LOGOUT
            </button>
          </div>
        </div>
      </div>

      {/* Main Layout */}
      <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 73px)', padding: 12, gap: 10 }}>

        {/* Session Info Bar */}
        {currentSession && (
          <div className="retro-window" style={{ flexShrink: 0 }}>
            <div style={{ padding: '5px 10px', display: 'flex', alignItems: 'center', gap: 8 }}>
              <User size={13} color="#6B8C42" />
              <span style={{ fontFamily: 'VT323, monospace', fontSize: 17, color: '#2b1d1d' }}>
                SESSION: {currentSession.session_token}
              </span>
              <span style={{ marginLeft: 'auto', fontFamily: 'VT323, monospace', fontSize: 16, color: isActive ? '#6B8C42' : '#a08060' }}>
                {isActive ? '[ACTIVE]' : '[ENDED]'}
              </span>
            </div>
          </div>
        )}

        {/* Chat History */}
        <CafeChatHistory
          messages={messages}
          emptyMessage="Belum ada pesan"
          emptySubmessage={currentSession ? 'Tunggu pesan dari customer' : 'Pilih atau buat session terlebih dahulu'}
        />

        {/* Input Section */}
        {isActive && (
          <div className="retro-window" style={{ flexShrink: 0 }}>
            <div className="retro-titlebar">
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className="retro-titlebar-icon" />
                <span className="retro-titlebar-title">staff_reply.exe</span>
              </div>
            </div>

            <div style={{ padding: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>

              {/* Mode Toggle */}
              <div style={{ display: 'flex', gap: 0, border: '2px solid #2b1d1d', width: 'fit-content' }}>
                <button
                  onClick={() => setIsVoiceMode(true)}
                  style={{
                    padding: '4px 14px',
                    fontFamily: 'VT323, monospace',
                    fontSize: 17,
                    background: isVoiceMode ? '#f26a3d' : '#f4dcc8',
                    color: isVoiceMode ? '#fff8f0' : '#2b1d1d',
                    border: 'none',
                    borderRight: '2px solid #2b1d1d',
                    cursor: 'pointer',
                    display: 'flex', alignItems: 'center', gap: 5,
                  }}
                >
                  <Mic size={13} />
                  VOICE
                </button>
                <button
                  onClick={() => setIsVoiceMode(false)}
                  style={{
                    padding: '4px 14px',
                    fontFamily: 'VT323, monospace',
                    fontSize: 17,
                    background: !isVoiceMode ? '#f26a3d' : '#f4dcc8',
                    color: !isVoiceMode ? '#fff8f0' : '#2b1d1d',
                    border: 'none',
                    cursor: 'pointer',
                    display: 'flex', alignItems: 'center', gap: 5,
                  }}
                >
                  TYPE
                </button>
              </div>

              {/* Input Area */}
              {isVoiceMode ? (
                <div style={{ display: 'flex', alignItems: 'stretch', gap: 8 }}>
                  <button
                    onClick={toggleRecording}
                    style={{
                      width: 46,
                      flexShrink: 0,
                      background: isRecording ? '#b85c4a' : '#efb36d',
                      border: '2px solid #2b1d1d',
                      boxShadow: isRecording ? 'none' : '3px 3px 0 #2b1d1d',
                      cursor: 'pointer',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      transform: isRecording ? 'translate(3px,3px)' : 'none',
                    }}
                  >
                    {isRecording ? <MicOff size={18} color="#fff8f0" /> : <Mic size={18} color="#2b1d1d" />}
                  </button>
                  <div
                    className="retro-inset"
                    style={{ flex: 1, fontFamily: 'VT323, monospace', fontSize: 18, color: currentMessage ? '#2b1d1d' : '#a08060', minHeight: 48, display: 'flex', alignItems: 'center' }}
                  >
                    {currentMessage || (isRecording ? '> MEREKAM...' : '> Klik mikrofon dan bicara...')}
                    {isRecording && <span className="retro-cursor" />}
                  </div>
                  <button
                    onClick={handleSend}
                    disabled={!currentMessage}
                    className="retro-btn-primary"
                    style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 46, flexShrink: 0, padding: 0 }}
                  >
                    <Send size={16} />
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', gap: 8 }}>
                  <textarea
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="> Ketik balasan..."
                    className="retro-textarea"
                    rows={2}
                    style={{ flex: 1 }}
                  />
                  <button
                    onClick={handleSend}
                    disabled={!currentMessage}
                    className="retro-btn-primary"
                    style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 46, flexShrink: 0, padding: 0 }}
                  >
                    <Send size={16} />
                  </button>
                </div>
              )}

              {/* Quick Responses */}
              <div>
                <p className="retro-label" style={{ marginBottom: 5, fontSize: 14, color: '#a08060' }}>
                  -- QUICK RESPONSES --
                </p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {[
                    'Baik, siap!',
                    'Mohon tunggu sebentar',
                    'Terima kasih atas pesanannya',
                    'Total Rp 25.000',
                    'Ada yang lain?',
                    'Promo hari ini diskon 10%',
                  ].map((response) => (
                    <button
                      key={response}
                      onClick={() => setCurrentMessage(response)}
                      className="retro-chip"
                    >
                      {response}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="retro-statusbar">
              <span>{currentMessage.length} chars</span>
              <span className="retro-statusbar-cell">Enter untuk kirim</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}