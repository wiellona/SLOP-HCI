const ML_SERVICE_URL = process.env.NEXT_PUBLIC_ML_SERVICE_URL || 'http://localhost:8000';

export class MLAPIClient {
  private ws: WebSocket | null = null;
  private sessionToken: string;
  
  constructor(sessionToken: string) {
    this.sessionToken = sessionToken;
  }
  
  async connect(
    onFrameUpdate: (data: any) => void,
    onPartialText: (text: string, confidence: number) => void,
    onFinalResult: (result: any) => void
  ): Promise<void> {
    const wsUrl = ML_SERVICE_URL.replace('http', 'ws');
    this.ws = new WebSocket(`${wsUrl}/v1/sign/stream`);
    
    return new Promise((resolve, reject) => {
      if (!this.ws) return reject();
      
      this.ws.onopen = () => {
        this.ws?.send(JSON.stringify({ session_token: this.sessionToken }));
        resolve();
      };
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          switch (data.event) {
            case 'inference_started':
              console.log('Inference started:', data.inference_id);
              break;
            case 'text_streamed':
              onFrameUpdate({
                bounding_box: data.bounding_box,
                occlusion_detected: data.occlusion_detected,
                frame_confidence: data.confidence_score
              });
              if (data.partial_text || data.translated_partial_text) {
                const partialText = data.translated_partial_text || data.partial_text;
                onPartialText(partialText, data.confidence_score);
              }
              break;
            case 'inference_complete':
              onFinalResult(data);
              break;
          }
        } catch (err) {
          console.error('Parse error:', err);
        }
      };
      
      this.ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        reject(err);
      };
    });
  }
  
  sendFrame(frameBase64: string): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      // Kirim sebagai binary untuk efisiensi
      const binaryData = atob(frameBase64.split(',')[1]);
      const bytes = new Uint8Array(binaryData.length);
      for (let i = 0; i < binaryData.length; i++) {
        bytes[i] = binaryData.charCodeAt(i);
      }
      this.ws.send(bytes);
    }
  }
  
  endSigning(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'end_signing' }));
    }
  }
  
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}