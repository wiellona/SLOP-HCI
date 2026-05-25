"use client";

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  is_occluded: boolean;
  occlusion_score: number;
}

interface BoundingBoxOverlayProps {
  boundingBox: BoundingBox | null;
  confidence: number;
  occlusionDetected: boolean;
}

export function BoundingBoxOverlay({
  boundingBox,
  confidence,
  occlusionDetected,
}: BoundingBoxOverlayProps) {
  if (!boundingBox) return null;

  const getBoxColor = () => {
    if (occlusionDetected) return "border-[#D4A843]";
    if (confidence >= 0.9) return "border-[#6B8C42]";
    if (confidence >= 0.7) return "border-[#D4A843]";
    return "border-[#B85C4A]";
  };

  const getBoxStyle = () => {
    const mirroredX = 1 - boundingBox.x - boundingBox.width;
    return {
      left: `${mirroredX * 100}%`,
      top: `${boundingBox.y * 100}%`,
      width: `${boundingBox.width * 100}%`,
      height: `${boundingBox.height * 100}%`,
    };
  };

  return (
    <div
      className={`absolute border-4 rounded-xl transition-all duration-200 pointer-events-none ${getBoxColor()}`}
      style={getBoxStyle()}
    >
      <div className="absolute -top-6 left-0 rounded-lg bg-black/60 px-2 py-0.5 text-xs text-white/90 whitespace-nowrap">
        Conf: {Math.round(confidence * 100)}%
      </div>
    </div>
  );
}
