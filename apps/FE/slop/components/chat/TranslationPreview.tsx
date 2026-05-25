"use client";

import { Edit3 } from "lucide-react";

interface CafeTranslationPreviewProps {
  translation: string;
  isProcessing: boolean;
  onEdit: () => void;
}

export function CafeTranslationPreview({
  translation,
  isProcessing,
  onEdit,
}: CafeTranslationPreviewProps) {
  return (
    <div className="retro-window">
      <div className="retro-titlebar">
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="retro-titlebar-icon" />
          <span className="retro-titlebar-title">translation_output.txt</span>
        </div>
        {translation && (
          <button
            className="retro-winctrl"
            onClick={onEdit}
            title="Edit terjemahan"
            style={{
              width: "auto",
              padding: "0 8px",
              display: "flex",
              alignItems: "center",
              gap: 4,
            }}
          >
            <Edit3 size={10} />
            EDIT
          </button>
        )}
      </div>

      <div className="retro-inset" style={{ minHeight: 70, margin: 8 }}>
        {translation ? (
          <p
            style={{
              fontFamily: "VT323, monospace",
              fontSize: 22,
              color: "#2b1d1d",
              margin: 0,
              lineHeight: 1.3,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {translation}
            <span className="retro-cursor" />
          </p>
        ) : (
          <p
            style={{
              fontFamily: "VT323, monospace",
              fontSize: 18,
              color: "#a08060",
              margin: 0,
            }}
          >
            {isProcessing
              ? "> MEMPROSES GESTURE..."
              : "> MENUNGGU INPUT GESTURE..."}
            <span className="retro-cursor" />
          </p>
        )}
      </div>

      {/* Progress bar shown when processing */}
      {isProcessing && (
        <div style={{ padding: "0 8px 8px" }}>
          <div className="retro-progress-track">
            <div
              className="retro-progress-fill"
              style={{
                width: "60%",
                animation: "progress-anim 1.2s ease-in-out infinite alternate",
              }}
            />
          </div>
          <style>{`
            @keyframes progress-anim {
              from { width: 20%; }
              to { width: 90%; }
            }
          `}</style>
        </div>
      )}
    </div>
  );
}
