import axios from "axios";

const API_BASE_URL = "http://localhost:4000/api/v1";
const WS_BASE_URL =
  process.env.NEXT_PUBLIC_WS_BASE_URL || "http://localhost:4000";

const toWebSocketBaseUrl = (baseUrl: string) => baseUrl.replace(/^http/, "ws");

export interface Conversation {
  id: string;
  session_token: string;
  status: "ACTIVE" | "ENDED";
  created_at: string;
  ended_at?: string;
}

export interface Message {
  id: string;
  text: string;
  sender_type: "CUSTOMER" | "STAFF";
  confidence?: number;
  created_at: string;
}

interface Staff {
  id: string;
  display_name: string;
  email: string;
  role: string;
}

class ApiClient {
  private token: string | null = null;
  private currentUser: Staff | null = null;

  constructor() {
    if (typeof window !== "undefined") {
      this.token = localStorage.getItem("staff_token");
      const user = localStorage.getItem("staff_user");
      if (user) this.currentUser = JSON.parse(user);
    }
  }

  private getHeaders() {
    return {
      "Content-Type": "application/json",
      ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}),
    };
  }

  async register(data: {
    id: string;
    email: string;
    password: string;
    display_name: string;
    role: string;
  }) {
    const users = JSON.parse(localStorage.getItem("staff_users") || "[]");

    // Check if user already exists
    if (users.find((u: any) => u.id === data.id)) {
      throw new Error("User already exists");
    }

    const newUser = {
      id: data.id,
      email: data.email,
      display_name: data.display_name,
      password: data.password,
      role: data.role,
      created_at: new Date().toISOString(),
    };

    users.push(newUser);
    localStorage.setItem("staff_users", JSON.stringify(users));

    // Auto login after registration
    this.token = `mock_token_${Date.now()}`;
    this.currentUser = {
      id: data.id,
      display_name: data.display_name,
      email: data.email,
      role: data.role,
    };

    localStorage.setItem("staff_token", this.token);
    localStorage.setItem("staff_user", JSON.stringify(this.currentUser));

    return { access_token: this.token, user: this.currentUser };
  }
  async login(data: { id: string; password: string }) {
    const users = JSON.parse(localStorage.getItem("staff_users") || "[]");
    const user = users.find(
      (u: any) => u.id === data.id && u.password === data.password,
    );

    if (!user) {
      throw new Error("Invalid credentials");
    }

    this.token = `mock_token_${Date.now()}`;
    this.currentUser = {
      id: user.id,
      display_name: user.display_name,
      email: user.email,
      role: user.role,
    };

    if (typeof window !== "undefined") {
      localStorage.setItem("staff_token", this.token);
      localStorage.setItem("staff_user", JSON.stringify(this.currentUser));
    }

    return { access_token: this.token, user: this.currentUser };
  }

  logout() {
    this.token = null;
    this.currentUser = null;
    if (typeof window !== "undefined") {
      localStorage.removeItem("staff_token");
      localStorage.removeItem("staff_user");
    }
  }

  getCurrentUser() {
    return this.currentUser;
  }

  // Client-side session management
  async createSession(): Promise<{
    session_id: string;
    session_token: string;
  }> {
    const sessions = JSON.parse(localStorage.getItem("chat_sessions") || "[]");
    const newSession = {
      id: `session_${Date.now()}`,
      session_token: `TOKEN_${Math.random().toString(36).substring(2, 15)}`,
      status: "ACTIVE",
      created_at: new Date().toISOString(),
      messages: [],
    };

    sessions.push(newSession);
    localStorage.setItem("chat_sessions", JSON.stringify(sessions));

    return {
      session_id: newSession.id,
      session_token: newSession.session_token,
    };
  }

  async getAllSessions(): Promise<Conversation[]> {
    const sessions = JSON.parse(localStorage.getItem("chat_sessions") || "[]");
    return sessions;
  }

  async getSession(sessionId: string): Promise<Conversation> {
    const sessions = JSON.parse(localStorage.getItem("chat_sessions") || "[]");
    const session = sessions.find((s: any) => s.id === sessionId);
    return session;
  }

  async getSessionMessages(sessionId: string): Promise<Message[]> {
    const sessions = JSON.parse(localStorage.getItem("chat_sessions") || "[]");
    const session = sessions.find((s: any) => s.id === sessionId);
    return session?.messages || [];
  }

  async sendMessage(
    sessionId: string,
    text: string,
    senderType: "CUSTOMER" | "STAFF",
    confidence?: number,
  ) {
    const sessions = JSON.parse(localStorage.getItem("chat_sessions") || "[]");
    const sessionIndex = sessions.findIndex((s: any) => s.id === sessionId);

    if (sessionIndex === -1) throw new Error("Session not found");

    const newMessage = {
      id: `msg_${Date.now()}`,
      text,
      sender_type: senderType,
      confidence,
      created_at: new Date().toISOString(),
    };

    sessions[sessionIndex].messages.push(newMessage);
    localStorage.setItem("chat_sessions", JSON.stringify(sessions));

    return newMessage;
  }

  async endSession(sessionId: string) {
    const sessions = JSON.parse(localStorage.getItem("chat_sessions") || "[]");
    const sessionIndex = sessions.findIndex((s: any) => s.id === sessionId);

    if (sessionIndex !== -1) {
      sessions[sessionIndex].status = "ENDED";
      sessions[sessionIndex].ended_at = new Date().toISOString();
      localStorage.setItem("chat_sessions", JSON.stringify(sessions));
    }

    return { success: true };
  }

  getWebSocketUrl(sessionId?: string): string {
    const wsBaseUrl = toWebSocketBaseUrl(WS_BASE_URL);
    return `${wsBaseUrl}?session_id=${sessionId}`;
  }
}

export const apiClient = new ApiClient();
