"use client";

import { RefObject } from "react";
import Webcam from "react-webcam";
import { VideoOff } from "lucide-react";

interface CafeCameraFeedProps {
  webcamRef: RefObject<Webcam>;
  isActive: boolean;
  onToggle: () => void;
  borderColor: string;
  backgroundColor: string;
  statusText: string;
  confidence?: number;
  isTracking?: boolean;
  children?: React.ReactNode;
}

export function CafeCameraFeed({
  webcamRef,
  isActive,
  onToggle,
  borderColor,
  backgroundColor,
  statusText,
  confidence,
  isTracking,
  children,
}: CafeCameraFeedProps) {
  return (
    <div
      className="retro-window"
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        height: "100%",
      }}
    >
      <div className="retro-titlebar">
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="retro-titlebar-icon" />
          <span className="retro-titlebar-title">camera_feed.exe</span>
        </div>
        <div className="retro-winctrls">
          <button className="retro-winctrl">-</button>
          <button className="retro-winctrl">+</button>
          <button
            className="retro-winctrl"
            onClick={onToggle}
            style={{ background: "#e05528", color: "#fff8f0" }}
            title={isActive ? "Matikan kamera" : "Hidupkan kamera"}
          >
            x
          </button>
        </div>
      </div>

      {/* Camera area */}
      <div
        className="retro-scanlines"
        style={{
          flex: 1,
          minHeight: 0,
          position: "relative",
          background: backgroundColor,
          border: "2px solid #2b1d1d",
          borderTop: "none",
          overflow: "hidden",
          outline: `3px solid ${borderColor}`,
          outlineOffset: -3,
        }}
      >
        {isActive ? (
          <>
            <div
              style={{
                width: "100%",
                height: "100%",
                transform: "scaleX(-1)",
              }}
            >
              <Webcam
                ref={webcamRef}
                audio={false}
                mirrored={false}
                screenshotFormat="image/jpeg"
                screenshotQuality={1}
                videoConstraints={{
                  facingMode: "user",
                  width: 1280,
                  height: 720,
                  frameRate: 30,
                }}
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  display: "block",
                }}
              />
            </div>
            {children}
          </>
        ) : (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              gap: 12,
            }}
          >
            <VideoOff size={36} color="#f26a3d" />
            <p
              style={{
                fontFamily: "VT323, monospace",
                fontSize: 20,
                color: "#f4b26b",
              }}
            >
              KAMERA MATI
            </p>
            <button className="retro-btn-primary" onClick={onToggle}>
              HIDUPKAN KAMERA
            </button>
          </div>
        )}

        {/* Retro HUD overlay */}
        {isActive && (
          <>
            {/* Top-left corner brackets */}
            <div
              style={{
                position: "absolute",
                top: 8,
                left: 8,
                width: 20,
                height: 20,
                borderTop: "2px solid #f26a3d",
                borderLeft: "2px solid #f26a3d",
                zIndex: 10,
              }}
            />
            <div
              style={{
                position: "absolute",
                top: 8,
                right: 8,
                width: 20,
                height: 20,
                borderTop: "2px solid #f26a3d",
                borderRight: "2px solid #f26a3d",
                zIndex: 10,
              }}
            />
            <div
              style={{
                position: "absolute",
                bottom: 36,
                left: 8,
                width: 20,
                height: 20,
                borderBottom: "2px solid #f26a3d",
                borderLeft: "2px solid #f26a3d",
                zIndex: 10,
              }}
            />
            <div
              style={{
                position: "absolute",
                bottom: 36,
                right: 8,
                width: 20,
                height: 20,
                borderBottom: "2px solid #f26a3d",
                borderRight: "2px solid #f26a3d",
                zIndex: 10,
              }}
            />

            {/* Status bar at bottom */}
            <div
              style={{
                position: "absolute",
                bottom: 0,
                left: 0,
                right: 0,
                background: "rgba(244,220,200,0.92)",
                borderTop: "2px solid #2b1d1d",
                padding: "3px 10px",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                zIndex: 10,
              }}
            >
              <span
                style={{
                  fontFamily: "VT323, monospace",
                  fontSize: 17,
                  color: borderColor,
                }}
              >
                {`Status: ${statusText}`}
              </span>
              {isTracking && confidence !== undefined && (
                <span
                  style={{
                    fontFamily: "VT323, monospace",
                    fontSize: 16,
                    color: "#2b1d1d",
                  }}
                >
                  CONF: {Math.round(confidence * 100)}%
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
