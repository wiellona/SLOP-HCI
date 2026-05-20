const WebSocket = require('ws');
const jwt = require('jsonwebtoken');

class WebSocketManager {
  constructor(server) {
    this.wss = new WebSocket.Server({ server });
    this.sessions = new Map(); // sessionId -> Set of WebSocket connections
    this.setupWebSocket();
  }

  setupWebSocket() {
    this.wss.on('connection', (ws, req) => {
      const url = new URL(req.url, `http://${req.headers.host}`);
      const sessionId = url.searchParams.get('session_id');
      
      if (!sessionId) {
        ws.close(1008, 'Session ID required');
        return;
      }

      // Store connection
      if (!this.sessions.has(sessionId)) {
        this.sessions.set(sessionId, new Set());
      }
      this.sessions.get(sessionId).add(ws);

      ws.on('message', async (data) => {
        try {
          const message = JSON.parse(data);
          
          switch (message.type) {
            case 'auth':
              // Verify JWT token
              const token = message.token;
              if (token) {
                try {
                  const decoded = jwt.verify(token, process.env.JWT_SECRET);
                  ws.user = decoded;
                  ws.send(JSON.stringify({ type: 'auth_success' }));
                } catch (err) {
                  ws.send(JSON.stringify({ type: 'auth_error', error: 'Invalid token' }));
                }
              }
              break;
              
            case 'message':
              // Broadcast message to all connections in the session
              this.broadcastToSession(sessionId, {
                type: 'message',
                data: message.data,
              });
              break;
              
            case 'typing':
              // Broadcast typing status
              this.broadcastToSession(sessionId, {
                type: 'typing',
                data: {
                  user: ws.user?.id || 'unknown',
                  is_typing: message.is_typing,
                },
              });
              break;
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      });

      ws.on('close', () => {
        // Remove connection
        const connections = this.sessions.get(sessionId);
        if (connections) {
          connections.delete(ws);
          if (connections.size === 0) {
            this.sessions.delete(sessionId);
          }
        }
      });
    });
  }

  broadcastToSession(sessionId, message) {
    const connections = this.sessions.get(sessionId);
    if (connections) {
      const messageStr = JSON.stringify(message);
      connections.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(messageStr);
        }
      });
    }
  }

  sendToSession(sessionId, message) {
    this.broadcastToSession(sessionId, message);
  }
}

module.exports = WebSocketManager;