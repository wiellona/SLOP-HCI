# SLOP — Sign Language Output Provider

## Smart Cashier System

### Project Requirements Document & Technical Execution Plan

---

| Field                | Value                  |
| -------------------- | ---------------------- |
| **Document Version** | 1.0.0                  |
| **Classification**   | Internal — Engineering |
| **Date**             | 21 April 2026          |

> _Bridging Communication Gaps Between Deaf and Hearing Communities_

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture & Monorepo Structure](#2-system-architecture--monorepo-structure)
3. [Database Schema (Prisma / PostgreSQL)](#3-database-schema-prisma--postgresql)
4. [Real-Time Interaction Logic](#4-real-time-interaction-logic)
5. [UI/UX & AI Feedback Guidelines](#5-uiux--ai-feedback-guidelines)
6. [API & WebSocket Contracts](#6-api--websocket-contracts)
7. [RLHF Pipeline & Model Improvement Loop](#7-rlhf-pipeline--model-improvement-loop)
8. [Security & Privacy Architecture](#8-security--privacy-architecture)
9. [Deployment Architecture](#9-deployment-architecture)
10. [Project Milestones & Acceptance Criteria](#10-project-milestones--acceptance-criteria)

---

## 1. Executive Summary

SLOP (Sign Language Output Provider) is a dual-screen, real-time communication system designed to eliminate communication barriers between Deaf and hearing individuals in retail and cafe environments. The system leverages AI-powered sign language recognition (Siformer) and voice transcription (Whisper) to provide seamless, bidirectional communication through a shared chat interface displayed across two dedicated terminal screens.

> **MISSION:** Empower Deaf customers and hearing staff to transact naturally — without interpreters, paper, or workarounds — through an intelligent, feedback-learning accessibility platform.

### 1.1 Core Value Proposition

- Real-time sign language inference with confidence-aware visual feedback
- Two-stage message validation ensuring accuracy before transmission
- Continuous RLHF (Reinforcement Learning from Human Feedback) loop for model improvement
- Fully accessible, WCAG 2.1 AA compliant dual-screen UI
- Privacy-first architecture with session-scoped data handling

### 1.2 Key Stakeholders

| Stakeholder        | Role           | Primary Concern                                 |
| ------------------ | -------------- | ----------------------------------------------- |
| Deaf Customers     | Primary User   | Sign recognition accuracy, low-latency feedback |
| Hearing Staff      | Secondary User | Readability, speed, minimal training overhead   |
| Store Managers     | Operator       | Uptime, analytics, compliance                   |
| ML Engineers       | Maintainers    | RLHF data quality, model versioning             |
| QA / Accessibility | Auditors       | WCAG compliance, edge-case coverage             |

---

## 2. System Architecture & Monorepo Structure

The project is structured as a Turborepo monorepo, separating concerns across three primary application workspaces and a shared packages layer. This enables independent deployment of each service while sharing configuration, types, and utilities.

> **PATTERN:** Event-driven microservices with a shared real-time bus (Socket.io). The ML service operates asynchronously and communicates inferences via Socket.io rooms scoped to each conversation session.

### 2.1 High-Level Architecture Tiers

| Tier               | Technology         | Responsibility                                        |
| ------------------ | ------------------ | ----------------------------------------------------- |
| Client Layer       | React / Next.js    | Dual-screen UI, camera capture, real-time rendering   |
| Application Layer  | Node.js / Express  | Auth, session mgmt, Socket.io orchestration, REST API |
| ML Inference Layer | Python / FastAPI   | Siformer sign inference, Whisper voice-to-text        |
| Data Layer         | PostgreSQL 15      | Persistent storage, RLHF logs, user profiles          |
| DevOps Layer       | Docker / Turborepo | Containerization, build orchestration, CI/CD          |

### 2.2 Monorepo Folder Hierarchy

The following structure enforces strict separation of concerns and enables independent CI/CD pipelines per workspace:

```
slop-monorepo/
├── apps/
│   ├── web/                         # Next.js 14 — Dual-Screen Frontend
│   │   ├── app/
│   │   │   ├── (staff)/             # Staff terminal route group
│   │   │   │   ├── layout.tsx
│   │   │   │   └── page.tsx         # Staff screen (voice input + chat log)
│   │   │   ├── (customer)/          # Customer terminal route group
│   │   │   │   ├── layout.tsx
│   │   │   │   └── page.tsx         # Customer screen (sign camera + preview)
│   │   │   ├── api/                 # Next.js API routes (thin BFF layer)
│   │   │   └── layout.tsx
│   │   ├── components/
│   │   │   ├── camera/              # CameraFeed, BoundingBoxOverlay
│   │   │   ├── chat/                # ChatBubble, ChatHistory, GhostText
│   │   │   ├── controls/            # SendButton, EditPanel, ModalOverlay
│   │   │   └── shared/              # AccessibilityBar, StatusIndicator
│   │   ├── hooks/                   # useSocket, useInference, useSession
│   │   ├── lib/                     # socket-client.ts, api-client.ts
│   │   ├── public/
│   │   ├── next.config.mjs
│   │   ├── tailwind.config.ts
│   │   └── tsconfig.json
│
│   ├── server/                      # Node.js Express + Socket.io Backend
│   │   ├── src/
│   │   │   ├── controllers/         # session.controller.ts, message.controller.ts
│   │   │   ├── middlewares/         # auth.middleware.ts, logger.middleware.ts
│   │   │   ├── routes/              # v1/sessions, v1/messages, v1/rlhf
│   │   │   ├── services/            # session.service.ts, rlhf.service.ts
│   │   │   ├── sockets/
│   │   │   │   ├── inference.handler.ts
│   │   │   │   └── message.handler.ts
│   │   │   ├── lib/
│   │   │   │   └── prisma.ts        # Prisma client singleton
│   │   │   ├── app.ts
│   │   │   └── server.ts
│   │   ├── prisma/
│   │   │   ├── schema.prisma
│   │   │   └── migrations/
│   │   ├── Dockerfile
│   │   ├── package.json
│   │   └── tsconfig.json
│
│   └── ml-service/                  # Python FastAPI — AI Inference Engine
│       ├── app/
│       │   ├── api/
│       │   │   └── v1/
│       │   │       ├── sign.py      # Siformer inference endpoint
│       │   │       ├── voice.py     # Whisper inference endpoint
│       │   │       └── rlhf.py      # Correction push-back endpoint
│       │   ├── core/
│       │   │   ├── config.py        # Env settings via pydantic-settings
│       │   │   └── events.py        # Startup / shutdown lifecycle
│       │   ├── models/
│       │   │   ├── siformer/        # Model weights + loader
│       │   │   └── whisper/         # Whisper model loader
│       │   ├── schemas/             # Pydantic request/response models
│       │   ├── services/
│       │   │   ├── sign_inference.py
│       │   │   ├── voice_inference.py
│       │   │   └── socketio_emitter.py
│       │   └── main.py
│       ├── tests/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── pyproject.toml
│
├── packages/
│   ├── shared-types/                # TypeScript types shared across apps
│   │   ├── src/
│   │   │   ├── socket-events.ts     # All socket event payloads
│   │   │   ├── api-contracts.ts     # REST request/response shapes
│   │   │   └── domain.ts            # User, Message, Session types
│   │   └── package.json
│   ├── ui-components/               # Shared headless + styled components
│   │   ├── src/
│   │   └── package.json
│   ├── eslint-config/               # Shared ESLint ruleset
│   └── tsconfig/                    # Base tsconfig.json configs
│
├── docker-compose.yml               # Local dev: all services
├── turbo.json                       # Turborepo pipeline config
├── package.json                     # Root workspace config
└── .env.example
```

### 2.3 Turborepo Pipeline Configuration

```json
{
  "$schema": "https://turbo.build/schema.json",
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": [".next/**", "dist/**", "__pycache__/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "lint": {},
    "test": {
      "dependsOn": ["^build"],
      "outputs": ["coverage/**"]
    },
    "db:migrate": {
      "cache": false
    }
  }
}
```

---

## 3. Database Schema (Prisma / PostgreSQL)

The database runs on a self-managed **PostgreSQL 15** instance accessed via Prisma ORM. Connection pooling is handled by **PgBouncer** in transaction mode. The schema is organized around four core entities.

> **DESIGN PRINCIPLE:** `AI_Inference_Logs` is the most critical table in the system. Its fidelity directly determines the quality of future fine-tuning. Every inference — whether accepted, edited, or rejected — must be persisted.

### 3.1 Prisma Schema Definition

```prisma
// apps/server/prisma/schema.prisma

generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")   // Points to PgBouncer (port 6432)
}

// ─── ENUMS ────────────────────────────────────────────────────────────────────

enum UserRole {
  STAFF
  CUSTOMER
  ADMIN
}

enum InputModality {
  SIGN
  VOICE
  TEXT
}

enum ConversationStatus {
  ACTIVE
  COMPLETED
  ABANDONED
}

enum MessageStatus {
  PREVIEW   // Stage 1: visible only to sender
  SENT      // Stage 2: synced to both screens
}

// ─── USERS ────────────────────────────────────────────────────────────────────

model User {
  id            String   @id @default(cuid())
  display_name  String
  email         String?  @unique
  password_hash String?  // Nullable for anonymous customer sessions

  role          UserRole @default(CUSTOMER)

  // Accessibility metadata
  is_deaf        Boolean @default(false)
  preferred_lang String  @default("en-US")
  sign_dialect   String? // e.g. 'ASL', 'BSL', 'BISINDO'

  // Terminal binding
  terminal_id String?  // Physical device identifier
  is_active   Boolean  @default(true)

  created_at DateTime @default(now())
  updated_at DateTime @updatedAt

  messages       Message[]
  conversations  ConversationParticipant[]
  inference_logs AI_Inference_Log[]

  @@map("users")
}

// ─── CONVERSATIONS ────────────────────────────────────────────────────────────

model Conversation {
  id            String             @id @default(cuid())
  session_token String             @unique @default(uuid())
  status        ConversationStatus @default(ACTIVE)

  // Terminal tracking
  staff_terminal    String? // Socket room ID for staff screen
  customer_terminal String? // Socket room ID for customer screen

  started_at DateTime  @default(now())
  ended_at   DateTime?

  messages       Message[]
  participants   ConversationParticipant[]
  inference_logs AI_Inference_Log[]

  @@map("conversations")
}

model ConversationParticipant {
  id              String       @id @default(cuid())
  conversation    Conversation @relation(fields: [conversation_id], references: [id])
  conversation_id String
  user            User         @relation(fields: [user_id], references: [id])
  user_id         String

  joined_at DateTime  @default(now())
  left_at   DateTime?

  @@unique([conversation_id, user_id])
  @@map("conversation_participants")
}

// ─── MESSAGES ─────────────────────────────────────────────────────────────────

model Message {
  id              String       @id @default(cuid())
  conversation    Conversation @relation(fields: [conversation_id], references: [id])
  conversation_id String
  sender          User         @relation(fields: [sender_id], references: [id])
  sender_id       String

  content  String        // Final, human-confirmed message text
  modality InputModality
  status   MessageStatus @default(PREVIEW)

  // Linked AI log (nullable — manually typed messages have no inference log)
  inference_log    AI_Inference_Log? @relation(fields: [inference_log_id], references: [id])
  inference_log_id String?           @unique

  created_at DateTime  @default(now())
  sent_at    DateTime? // Set when status transitions to SENT

  @@map("messages")
}

// ─── AI INFERENCE LOGS (RLHF CORE) ───────────────────────────────────────────

model AI_Inference_Log {
  id              String       @id @default(cuid())
  conversation    Conversation @relation(fields: [conversation_id], references: [id])
  conversation_id String
  user            User         @relation(fields: [user_id], references: [id])
  user_id         String

  // AI Output
  raw_prediction_text String        // What the model predicted
  confidence_score    Float         // 0.0 – 1.0 (normalised)
  model_version       String        // e.g. 'siformer-v2.1', 'whisper-large-v3'
  input_modality      InputModality

  // Human Correction (RLHF Signal)
  final_corrected_text String?  // What the user actually sent after editing
  was_edited           Boolean  @default(false)
  was_rejected         Boolean  @default(false) // User discarded the prediction entirely
  edit_delta_chars     Int?     // Levenshtein distance — computed server-side on finalization

  // Contextual metadata
  occlusion_detected   Boolean @default(false)
  frame_rate_avg       Float?  // Avg FPS during the capture window
  inference_latency_ms Int?    // End-to-end inference time in milliseconds

  // RLHF export tracking
  exported_for_training Boolean   @default(false)
  exported_at           DateTime?

  created_at DateTime @default(now())
  message    Message?

  @@index([conversation_id, created_at])
  @@index([input_modality, was_edited])
  @@index([exported_for_training])
  @@map("ai_inference_logs")
}
```

### 3.2 Schema Design Rationale

| Design Decision                   | Rationale                                                                                                                                                                                |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MessageStatus` (PREVIEW vs SENT) | Enforces Two-Stage Validation at the data layer. A message exists in the DB as `PREVIEW` without being visible to the receiver, then atomically promoted to `SENT` on user confirmation. |
| `AI_Inference_Log.was_edited`     | The primary RLHF signal. A high edit rate on a given `model_version` triggers a retraining flag in the ML pipeline.                                                                      |
| `edit_delta_chars` (Levenshtein)  | Quantifies correction severity. Small deltas may be typo fixes; large deltas indicate model failure cases warranting review.                                                             |
| `model_version` field             | Enables A/B analysis across model versions — critical for evaluating fine-tuning improvements over time.                                                                                 |
| `exported_for_training` flag      | Prevents duplicate training data. The RLHF export pipeline queries `WHERE exported_for_training = false` and marks records after export.                                                 |
| `password_hash` nullable          | Customer terminals may operate as anonymous sessions scoped to a single conversation, avoiding friction for Deaf users who just want to order.                                           |

### 3.3 Database Infrastructure

- **Engine:** PostgreSQL 15
- **Connection Pooling:** PgBouncer in transaction mode (port `6432`)
- **Migrations:** Managed via `prisma migrate deploy` in CI/CD pipeline
- **Direct URL:** `DIRECT_DATABASE_URL` bypasses PgBouncer for Prisma migration runner only
- **Backups:** Continuous WAL archiving + daily `pg_dump` snapshots to object storage

```sql
-- Run once on a fresh database instance
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- for @default(uuid())
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- for future fuzzy search on messages
```

---

## 4. Real-Time Interaction Logic

The core interaction model follows a **Two-Stage Validation** workflow that protects against premature transmission of inaccurate AI predictions. This design directly addresses the Human Factors "Gulf of Evaluation" by ensuring users have full agency over AI-generated content before it reaches the receiver.

### 4.1 Two-Stage Validation Workflow

**Stage 1 — LOCAL PREVIEW**
Inference text appears in the sender's preview pane with an Edit affordance. It is **not** yet visible to the receiver. The sender can accept, edit, or discard. The message record exists in the DB with `status = PREVIEW`.

**Stage 2 — GLOBAL SYNC**
Only when the sender clicks "Send" does the message atomically update to `status = SENT` and broadcast to the shared conversation history on both screens.

### 4.2 Message State Machine

```
INFERRING          // AI processing active, ghost text streaming
     │
     ▼
PREVIEW_READY      // Inference complete, sender sees result + Edit button
     │
     ├── [EDIT]──► EDITING       // Sender modifies text in-place
     │                │
     │                └──► PREVIEW_READY  (loop until satisfied)
     │
     ├── [DISCARD]──► IDLE       // DB log: was_rejected = true
     │
     └── [SEND]──────────────────────────────────►
                                                  │
                                                  ▼
                                            SENDING        // Optimistic UI update
                                                  │
                                                  ▼
                                            DELIVERED      // Socket confirms global sync
                                                  │
                                                  ▼
                                            IDLE           // Ready for next input
```

### 4.3 Socket.io Room Architecture

Each conversation session maps to a dedicated Socket.io room scoped by `session_token`. Both terminal screens join the same room, enabling selective event broadcasting.

```
Room:                session:{session_token}
Sub-room (customer): session:{session_token}:customer
Sub-room (staff):    session:{session_token}:staff

Rules:
  - ML Service emits to sender sub-room only     → Stage 1 (private preview)
  - 'message_finalized' emits to full room       → Stage 2 (public sync)
  - Both terminals subscribe to full room on connect
```

### 4.4 Event Flow — Sign Language Path

| #   | Event / Action       | Description                                                                        |
| --- | -------------------- | ---------------------------------------------------------------------------------- |
| 1   | Camera stream starts | Browser `MediaStream` API captures 30fps video on customer terminal                |
| 2   | Frames sent to ML    | WebSocket stream or chunked base64 frames sent to FastAPI `/v1/sign/stream`        |
| 3   | `inference_started`  | ML emits socket event; UI renders bounding box and ghost text placeholder          |
| 4   | `text_streamed`      | Partial token predictions stream back; ghost text updates in real-time             |
| 5   | `inference_complete` | Full prediction + confidence score returned; preview pane activates                |
| 6   | User validates       | Sender accepts or edits; Stage 1 complete                                          |
| 7   | `message_finalized`  | Send clicked; server updates message to `SENT`, broadcasts to room                 |
| 8   | RLHF log written     | Server persists `AI_Inference_Log` with `raw_prediction` vs `final_corrected_text` |

### 4.5 Event Flow — Voice Path

| #   | Event / Action          | Description                                                   |
| --- | ----------------------- | ------------------------------------------------------------- |
| 1   | Mic capture starts      | Staff terminal captures audio via `MediaRecorder` API         |
| 2   | Audio chunk posted      | 500ms audio chunks `POST`ed to FastAPI `/v1/voice/transcribe` |
| 3   | `inference_started`     | Server emits event to staff sub-room                          |
| 4   | `text_streamed`         | Whisper partial transcript streams back token-by-token        |
| 5   | `inference_complete`    | Final transcript returned with confidence score               |
| 6   | Staff validates & sends | Same Two-Stage flow applies as sign path                      |

---

## 5. UI/UX & AI Feedback Guidelines

The UI is designed to reduce cognitive load and address both the Gulf of Execution (what can I do?) and Gulf of Evaluation (what just happened?) for both Deaf and hearing users.

### 5.1 Dual-Screen Layout Philosophy

| Customer Terminal (Deaf User)         | Staff Terminal (Hearing User)                     |
| ------------------------------------- | ------------------------------------------------- |
| Large camera feed (top 60% of screen) | Text input area (voice or keyboard)               |
| Bounding box overlay on camera feed   | Real-time transcription preview                   |
| Ghost text preview below camera       | Chat history (mirrored conversation)              |
| Edit / Send / Discard controls        | Notification badge for new incoming messages      |
| Scrollable chat history (bottom 40%)  | Confidence indicator for incoming signed messages |

### 5.2 Dynamic Bounding Box — Gulf of Evaluation

The bounding box surrounding the detected hand/body region communicates AI confidence in real-time through color and animation:

| State           | Visual Cue                            | Trigger Condition                          |
| --------------- | ------------------------------------- | ------------------------------------------ |
| HIGH CONFIDENCE | `#22C55E` Solid Green                 | `confidence_score > 0.90` and no occlusion |
| LOW CONFIDENCE  | `#EF4444` Solid Red + pulse animation | `confidence_score <= 0.90`                 |
| OCCLUSION       | `#F59E0B` Dashed Amber + warning icon | Hand/body partially outside frame          |
| PROCESSING      | `#0EA5E9` Animated blue scanning line | Inference in-progress state                |
| NO DETECTION    | No box rendered                       | No human landmarks detected in frame       |

```typescript
// components/camera/BoundingBoxOverlay.tsx

const getBoxStyle = (
  score: number,
  occlusion: boolean,
  state: InferenceState,
) => {
  if (state === "PROCESSING")
    return { color: "#0EA5E9", animated: true, dash: false };
  if (occlusion) return { color: "#F59E0B", animated: false, dash: true };
  if (score > 0.9) return { color: "#22C55E", animated: false, dash: false };
  return { color: "#EF4444", animated: true, dash: false };
};
```

### 5.3 Ghost Text (Real-Time Inference Preview)

As the ML service streams partial predictions via `text_streamed` socket events, a "ghost text" element renders beneath the camera feed — styled in muted italics to clearly distinguish AI prediction from confirmed user content.

- Ghost text appears ~200ms after `inference_started` fires
- Text updates token-by-token as `text_streamed` events arrive
- On `inference_complete`, ghost text transitions to the editable preview pane
- If `confidence_score < 0.90`, ghost text is decorated with a red underline
- Ghost text is non-interactive — pointer events are disabled on it

> **ACCESSIBILITY NOTE:** Ghost text minimum font size is 24px. High contrast mode is supported via `prefers-contrast` media query. Screen readers receive `aria-live="polite"` announcements as tokens stream in.

### 5.4 Edit Mode UI Specification

- Edit button appears in Stage 1 preview state, styled as a secondary CTA
- Clicking Edit transitions preview text to a `<textarea>` with auto-focus
- Live character count displayed; changes beyond 50% of original text length trigger a soft warning toast
- "Confirm Edit" button promotes changes back to preview state
- All edits are diffed against `raw_prediction_text` via Levenshtein distance, stored as `edit_delta_chars` on finalization

### 5.5 Accessibility Requirements (WCAG 2.1 AA)

| Requirement             | Implementation                                                               |
| ----------------------- | ---------------------------------------------------------------------------- |
| Color contrast ≥ 4.5:1  | All text/background pairs validated via `axe-core` in CI                     |
| Keyboard navigation     | All interactive elements reachable via Tab; focus ring always visible        |
| Touch targets ≥ 44×44px | All buttons and controls meet minimum tap target size                        |
| Motion reduction        | Bounding box animations respect `prefers-reduced-motion` media query         |
| Screen reader support   | ARIA labels on all controls; `aria-live` regions for dynamic content updates |

---

## 6. API & WebSocket Contracts

All socket events use a consistent envelope format. TypeScript payload types are defined in `packages/shared-types` and imported by both `apps/web` and `apps/server`.

### 6.1 Socket Event Contracts

#### 6.1.1 `inference_started`

```typescript
// Emitter:  ML Service → Node Server → Client
// Channel:  session:{session_token}:{sender_terminal}  (Stage 1 — sender only)

interface InferenceStartedPayload {
  event: "inference_started";
  session_token: string;
  inference_id: string; // Temp ID linking all events in this inference chain
  modality: "SIGN" | "VOICE";
  timestamp: string; // ISO 8601
}
```

#### 6.1.2 `text_streamed`

```typescript
// Emitter:  ML Service → Node Server → Client
// Channel:  session:{session_token}:{sender_terminal}
// Frequency: ~100ms intervals or per decoded token

interface TextStreamedPayload {
  event: "text_streamed";
  inference_id: string;
  partial_text: string; // Accumulating predicted text so far
  confidence_score: number; // Latest frame confidence score (0.0 – 1.0)
  occlusion_detected: boolean;
  bounding_box: {
    x: number;
    y: number;
    width: number;
    height: number; // Normalised 0–1 viewport coordinates
  } | null;
}
```

#### 6.1.3 `inference_complete`

```typescript
// Emitter:  ML Service → Node Server → Client
// Channel:  session:{session_token}:{sender_terminal}

interface InferenceCompletePayload {
  event: "inference_complete";
  inference_id: string;
  raw_prediction_text: string;
  final_confidence_score: number;
  model_version: string;
  inference_latency_ms: number;
  occlusion_detected: boolean;
}
```

#### 6.1.4 `message_finalized`

```typescript
// Emitter:  Node Server → ALL clients in session room
// Channel:  session:{session_token}
// Trigger:  User clicks 'Send' — Stage 2 complete

interface MessageFinalizedPayload {
  event: "message_finalized";
  message_id: string; // Persisted DB message ID
  conversation_id: string;
  sender_id: string;
  content: string; // Final confirmed text
  modality: "SIGN" | "VOICE" | "TEXT";
  sent_at: string; // ISO 8601
}
```

#### 6.1.5 `session_ended`

```typescript
// Emitter:  Node Server → ALL clients
// Channel:  session:{session_token}
// Trigger:  Staff ends session or inactivity timeout

interface SessionEndedPayload {
  event: "session_ended";
  session_token: string;
  reason: "STAFF_ENDED" | "TIMEOUT" | "ERROR";
  ended_at: string;
}
```

### 6.2 REST API Contracts

#### 6.2.1 Session Management

| Method  | Endpoint                        | Auth      | Description                                              |
| ------- | ------------------------------- | --------- | -------------------------------------------------------- |
| `POST`  | `/api/v1/sessions`              | Staff JWT | Create new conversation session, returns `session_token` |
| `GET`   | `/api/v1/sessions/:id`          | Any JWT   | Fetch session details and participant list               |
| `PATCH` | `/api/v1/sessions/:id/end`      | Staff JWT | Mark session `COMPLETED`, trigger cleanup                |
| `GET`   | `/api/v1/sessions/:id/messages` | Any JWT   | Paginated message history for session                    |

#### 6.2.2 RLHF Correction Push-Back (ML Service → Node Server)

This endpoint is called internally by the ML service after a message is finalized, persisting the correction signal for future training export.

```
POST /api/v1/rlhf/corrections

Headers:
  Authorization: Bearer <ML_SERVICE_API_KEY>
  Content-Type: application/json

Request Body:
{
  "inference_id":          "string",   // Links to AI_Inference_Log record
  "raw_prediction_text":   "string",
  "final_corrected_text":  "string | null",
  "was_edited":            "boolean",
  "was_rejected":          "boolean",
  "confidence_score":      "number",
  "model_version":         "string",
  "input_modality":        "SIGN | VOICE",
  "inference_latency_ms":  "number",
  "occlusion_detected":    "boolean",
  "frame_rate_avg":        "number | null"
}

Response 201 Created:
{
  "log_id":            "string",
  "edit_delta_chars":  "number",  // Levenshtein distance, computed server-side
  "status":            "recorded"
}

Response 400 Bad Request:
{
  "error":   "INFERENCE_NOT_FOUND | VALIDATION_ERROR",
  "message": "string"
}
```

#### 6.2.3 RLHF Export

```
GET /api/v1/rlhf/export
  ?modality=SIGN
  &since=2025-01-01
  &tier=GOLD,SILVER,BRONZE
  &limit=1000

Response 200 OK:
{
  "records": [
    {
      "input_modality":    "SIGN",
      "raw_prediction":    "I want coffee",
      "corrected_text":    "I would like a coffee please",
      "confidence_score":  0.82,
      "edit_delta_chars":  18,
      "model_version":     "siformer-v2.1",
      "quality_tier":      "BRONZE"
    }
  ],
  "total":       1000,
  "exported_at": "2025-04-20T10:00:00Z"
}
```

#### 6.2.4 ML Inference Endpoints (FastAPI — `apps/ml-service`)

| Method | Endpoint               | Description                                                  |
| ------ | ---------------------- | ------------------------------------------------------------ |
| `POST` | `/v1/sign/infer`       | Single-frame sign inference, returns prediction + confidence |
| `WS`   | `/v1/sign/stream`      | WebSocket for real-time video frame streaming                |
| `POST` | `/v1/voice/transcribe` | Whisper transcription of an uploaded audio chunk             |
| `GET`  | `/v1/health`           | Service health, model load status, GPU availability          |
| `GET`  | `/v1/models`           | Lists loaded models and their current versions               |

---

## 7. RLHF Pipeline & Model Improvement Loop

The `ai_inference_logs` table is the sole source of truth for continuous model improvement. The pipeline follows a three-phase cycle: Collection, Export & Processing, and Fine-tuning.

### 7.1 Data Quality Tiers

| Tier       | Criteria                                    | Training Value                                              |
| ---------- | ------------------------------------------- | ----------------------------------------------------------- |
| **GOLD**   | `was_edited=true`, `edit_delta_chars < 10`  | Highest — near-correct predictions with minor surface fixes |
| **SILVER** | `was_edited=false` (accepted as-is)         | Positive reinforcement — model prediction was accurate      |
| **BRONZE** | `was_edited=true`, `edit_delta_chars >= 10` | Learning signal — significant human correction needed       |
| **SKIP**   | `was_rejected=true`                         | Excluded — too noisy to determine a reliable correct label  |

### 7.2 Pipeline Phases

**Phase 1 — Collection (Continuous)**
Every inference event is persisted in real-time. No sampling. The `exported_for_training` flag ensures idempotent export runs with no duplicate data.

**Phase 2 — Export & Processing (Scheduled: nightly cron)**

```bash
# 1. Fetch unexported records via the export endpoint
GET /api/v1/rlhf/export?tier=GOLD,SILVER,BRONZE&since=<last_run_timestamp>

# 2. Post-processing steps (run by ML pipeline script):
#    a. Verify edit_delta_chars is populated (compute if null)
#    b. Strip user_id — anonymize before leaving the system
#    c. Convert to model training format (frame sequence + corrected label)
#    d. Append to training data store (e.g. S3 bucket / NFS share)
#    e. Mark records exported_for_training = true via PATCH /api/v1/rlhf/mark-exported
```

**Phase 3 — Fine-tuning (Auto-triggered on threshold)**
Fine-tuning is triggered when either condition is met:

- More than **500 new GOLD/BRONZE records** have accumulated since the last training run, OR
- The rolling 7-day average `edit_delta_chars` for any `model_version` exceeds **15**

### 7.3 Model Versioning Policy

- Each fine-tuned model receives a semantic version bump: `siformer-v{MAJOR}.{MINOR}`
- New versions are shadow-deployed and A/B tested against 10% of live traffic before full rollout
- The `model_version` field in `ai_inference_logs` enables per-version performance analytics over time

---

## 8. Security & Privacy Architecture

### 8.1 Authentication Model

- **Staff terminals** use JWT-based auth with short-lived access tokens (15 min) and refresh tokens (7 days) stored in HTTP-only cookies
- **Customer terminals** may operate as anonymous sessions scoped to a single `Conversation` — no account required
- **ML service** authenticates to the Node server via a shared `ML_SERVICE_API_KEY` with IP allowlisting at the network layer
- All inter-service communication enforces HTTPS/WSS in production

```
JWT Payload:
{
  "sub":         "user_id",
  "role":        "STAFF | CUSTOMER | ADMIN",
  "terminal_id": "device_identifier",
  "iat":         1714000000,
  "exp":         1714000900
}
```

### 8.2 Data Privacy Controls

- Camera frames are **never persisted** — only inference text outputs are stored
- Audio buffers are deleted immediately after Whisper transcription completes
- Conversation data is retained for **90 days** by default (configurable via env var)
- RLHF exports strip `user_id` before leaving the application — only modality, prediction text, and correction are exported
- `password_hash` uses `bcrypt` with work factor 12

> **GDPR NOTE:** The system does not process biometric data in the legal sense. Skeletal landmark coordinates used for sign inference are computed in-memory and never written to disk. Legal review is recommended per deployment jurisdiction.

### 8.3 PostgreSQL Security Hardening

```sql
-- Dedicated application role with minimum required privileges
CREATE USER slop_app WITH PASSWORD '<strong_generated_password>';
GRANT CONNECT ON DATABASE slop TO slop_app;
GRANT USAGE ON SCHEMA public TO slop_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO slop_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO slop_app;

-- Read-only role for the RLHF export service
CREATE USER slop_rlhf_reader WITH PASSWORD '<strong_generated_password>';
GRANT CONNECT ON DATABASE slop TO slop_rlhf_reader;
GRANT SELECT ON ai_inference_logs TO slop_rlhf_reader;

-- Revoke public schema creation
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
```

---

## 9. Deployment Architecture

### 9.1 Docker Compose (Local Development)

```yaml
# docker-compose.yml
version: "3.9"

services:
  web:
    build: ./apps/web
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_SOCKET_URL=http://server:4000
    depends_on:
      - server

  server:
    build: ./apps/server
    ports:
      - "4000:4000"
    environment:
      - DATABASE_URL=postgresql://slop_app:${POSTGRES_PASSWORD}@pgbouncer:6432/slop_dev
      - DIRECT_DATABASE_URL=postgresql://slop_app:${POSTGRES_PASSWORD}@db:5432/slop_dev
      - ML_SERVICE_URL=http://ml-service:8000
      - ML_SERVICE_API_KEY=${ML_SERVICE_API_KEY}
      - JWT_SECRET=${JWT_SECRET}
      - JWT_REFRESH_SECRET=${JWT_REFRESH_SECRET}
    depends_on:
      - pgbouncer
      - ml-service

  ml-service:
    build: ./apps/ml-service
    ports:
      - "8000:8000"
    runtime: nvidia # GPU passthrough for inference
    environment:
      - MODEL_DEVICE=cuda
      - NODE_SERVER_URL=http://server:4000
      - ML_SERVICE_API_KEY=${ML_SERVICE_API_KEY}
    volumes:
      - ./models:/app/models:ro

  db:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: slop_dev
      POSTGRES_USER: slop_app
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  pgbouncer:
    image: edoburu/pgbouncer:latest
    ports:
      - "6432:6432"
    environment:
      DATABASE_URL: postgresql://slop_app:${POSTGRES_PASSWORD}@db:5432/slop_dev
      POOL_MODE: transaction
      MAX_CLIENT_CONN: 100
      DEFAULT_POOL_SIZE: 20
    depends_on:
      - db

volumes:
  postgres_data:
```

### 9.2 Environment Variables

| Variable                       | Description                                                       |
| ------------------------------ | ----------------------------------------------------------------- |
| `DATABASE_URL`                 | PostgreSQL connection string via PgBouncer (port `6432`)          |
| `DIRECT_DATABASE_URL`          | Direct PostgreSQL URL — used by Prisma migration runner only      |
| `POSTGRES_PASSWORD`            | PostgreSQL `slop_app` user password                               |
| `ML_SERVICE_URL`               | Internal URL of FastAPI inference service                         |
| `ML_SERVICE_API_KEY`           | Shared secret for ML → Node server authentication                 |
| `JWT_SECRET`                   | Secret key for signing access tokens                              |
| `JWT_REFRESH_SECRET`           | Separate secret key for refresh tokens                            |
| `NEXT_PUBLIC_SOCKET_URL`       | Public Socket.io server URL consumed by browser client            |
| `SIFORMER_MODEL_PATH`          | Path to Siformer model weights file                               |
| `WHISPER_MODEL_SIZE`           | Whisper model variant: `tiny`, `base`, `small`, `medium`, `large` |
| `MODEL_DEVICE`                 | Inference device: `cuda` or `cpu`                                 |
| `SESSION_INACTIVITY_TIMEOUT_S` | Seconds before auto-ending an inactive session (default: `300`)   |
| `RLHF_FINE_TUNE_THRESHOLD`     | New record count that triggers a fine-tune run (default: `500`)   |

### 9.3 Production Deployment Checklist

- [ ] PostgreSQL 15 provisioned with WAL archiving enabled
- [ ] PgBouncer configured in transaction mode, `MAX_CLIENT_CONN` tuned to DB max_connections
- [ ] SSL/TLS certificates provisioned for all public endpoints
- [ ] `DIRECT_DATABASE_URL` configured for migration runner only; app uses PgBouncer URL
- [ ] GPU node provisioned for `ml-service` (minimum: NVIDIA T4 or equivalent)
- [ ] Health check endpoints confirmed: `/api/v1/health` (Node), `/v1/health` (FastAPI)
- [ ] Log aggregation configured (e.g. Loki + Grafana, or ELK stack)
- [ ] Backup cron jobs verified: daily `pg_dump` + continuous WAL streaming to object storage
- [ ] `ML_SERVICE_API_KEY` and `JWT_SECRET` stored in a secrets manager, never committed to version control
- [ ] Prisma migrations run via `prisma migrate deploy` in CI/CD before service startup

---

## 10. Project Milestones & Acceptance Criteria

| Phase  | Milestone                 | Acceptance Criteria                                                                   | Est. Duration |
| ------ | ------------------------- | ------------------------------------------------------------------------------------- | ------------- |
| **M1** | Infrastructure & Schema   | Monorepo builds, DB migrations run cleanly, Docker Compose all services healthy       | 1 Week        |
| **M2** | Socket.io Real-Time Sync  | Message sent from Terminal A appears on Terminal B within 200ms                       | 1 Week        |
| **M3** | Two-Stage Validation UI   | Stage 1 preview visible only to sender; Stage 2 syncs to both screens atomically      | 1 Week        |
| **M4** | ML Service Integration    | Siformer inference returns prediction to client via socket with bounding box rendered | 2 Weeks       |
| **M5** | RLHF Logging              | All inferences written to `ai_inference_logs` with corrections captured on send       | 1 Week        |
| **M6** | Bounding Box + Ghost Text | Confidence-colored box renders correctly; ghost text streams token-by-token           | 1 Week        |
| **M7** | UAT & Accessibility Audit | WCAG 2.1 AA pass via `axe-core`; Deaf user group testing session complete             | 2 Weeks       |
| **M8** | Production Deploy         | Zero-downtime deploy, all health checks green, monitoring dashboards live             | 1 Week        |

**Total Estimated Timeline:** 10 Weeks from project kickoff to production-ready deployment.
Assumes a team of 3–4 engineers: 1 full-stack, 1 ML engineer, 1 frontend, 1 DevOps/QA.

### Definition of Done (per milestone)

- All acceptance criteria met and signed off by project lead
- Unit test coverage ≥ 80% for changed modules
- No open P0/P1 bugs
- Documentation updated in `/docs`
- PR peer-reviewed and merged to `main`

---

## Appendix A — Technology Decisions Log

| Decision           | Chosen        | Alternatives Considered        | Rationale                                                            |
| ------------------ | ------------- | ------------------------------ | -------------------------------------------------------------------- |
| Monorepo tool      | Turborepo     | Nx, Lerna                      | Best-in-class remote caching; native Next.js support                 |
| ORM                | Prisma        | TypeORM, Drizzle               | Type-safe client generation; excellent migration tooling             |
| Database           | PostgreSQL 15 | MySQL, SQLite                  | ACID compliance; JSONB support; open source; self-hostable           |
| Connection pooling | PgBouncer     | pgpool-II, built-in pooling    | Lightweight; transaction mode compatible with Prisma                 |
| Real-time layer    | Socket.io     | Raw WebSockets, SSE            | Room abstraction; automatic fallback transports; broad support       |
| Sign model         | Siformer      | MediaPipe + custom LSTM        | State-of-the-art transformer architecture for continuous SLR         |
| Voice model        | Whisper       | DeepSpeech, Vosk               | Best multilingual accuracy; fully self-hostable; active development  |
| ML framework       | FastAPI       | Flask, Django                  | Async-native; Pydantic integration; auto-generated OpenAPI docs      |
| CSS framework      | Tailwind CSS  | CSS Modules, styled-components | Utility-first; zero runtime overhead; great accessibility primitives |

---

_End of Document — SLOP PRD v1.0.0_

_Proprietary & Confidential_
