import { io, Socket } from 'socket.io-client';

const SOCKET_URL = process.env.NEXT_PUBLIC_SOCKET_URL || 'http://localhost:3001';

class SocketClient {
  private socket: Socket | null = null;
  private sessionId: string | null = null;

  connect(sessionId: string) {
    if (this.socket?.connected && this.sessionId === sessionId) {
      return this.socket;
    }
    
    this.sessionId = sessionId;
    this.socket = io(SOCKET_URL, {
      query: { sessionId },
      transports: ['websocket'],
    });
    
    this.socket.on('connect', () => {
      console.log('Socket connected:', this.socket?.id);
    });
    
    this.socket.on('disconnect', () => {
      console.log('Socket disconnected');
    });
    
    return this.socket;
  }
  
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.sessionId = null;
    }
  }
  
  getSocket() {
    return this.socket;
  }
  
  emit(event: string, data: any) {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    }
  }
  
  on(event: string, callback: (...args: any[]) => void) {
    if (this.socket) {
      this.socket.on(event, callback);
    }
  }
  
  off(event: string, callback?: (...args: any[]) => void) {
    if (this.socket) {
      this.socket.off(event, callback);
    }
  }
}

export const socketClient = new SocketClient();