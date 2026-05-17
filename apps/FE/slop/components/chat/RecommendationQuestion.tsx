'use client';

import { Coffee, ShoppingBag, Smile, ThumbsUp, HelpCircle } from 'lucide-react';

interface Question {
  icon: React.ElementType;
  text: string;
}

interface CafeRecommendationQuestionsProps {
  onSelectQuestion: (question: string) => void;
}

const QUESTIONS: Question[] = [
  { icon: Coffee, text: 'Saya mau pesan kopi' },
  { icon: Coffee, text: 'Saya mau pesan teh' },
  { icon: Coffee, text: 'Satu es kopi susu' },
  { icon: Coffee, text: 'Kopi tanpa gula' },
  { icon: ShoppingBag, text: 'Ada rekomendasi menu' },
  { icon: Smile, text: 'Terima kasih' },
  { icon: ThumbsUp, text: 'Saya setuju' },
  { icon: HelpCircle, text: 'Berapa total harganya' },
  { icon: HelpCircle, text: 'Bisa pesan antar' },
];

export function CafeRecommendationQuestions({ onSelectQuestion }: CafeRecommendationQuestionsProps) {
  return (
    <div className="retro-window">
      <div className="retro-titlebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="retro-titlebar-icon" />
          <span className="retro-titlebar-title">quick_phrases.dat</span>
        </div>
      </div>
      <div style={{ padding: 8 }}>
        <p style={{ fontFamily: 'VT323, monospace', fontSize: 15, color: '#a08060', margin: '0 0 6px', textTransform: 'uppercase', letterSpacing: 1 }}>
          -- Rekomendasi Pertanyaan --
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {QUESTIONS.map((q, idx) => (
            <button
              key={idx}
              onClick={() => onSelectQuestion(q.text)}
              className="retro-chip"
            >
              <q.icon size={12} />
              {q.text}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
