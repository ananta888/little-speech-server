import { Injectable, signal } from '@angular/core';
import { Engine } from './models';

export interface StreamHandlers {
  onReady?: () => void;
  onPartial?: (text: string) => void;
  onFinal?: (segment: { text: string; start: number; end: number; words?: unknown[] }) => void;
  onError?: (message: string) => void;
  onEnd?: () => void;
}

type StreamState = 'idle' | 'connecting' | 'ready' | 'recording' | 'stopping';

const TARGET_RATE = 16_000;

@Injectable({ providedIn: 'root' })
export class StreamService {
  readonly state = signal<StreamState>('idle');
  /** Anzahl der verworfenen Frames, weil der WebSocket-Puffer voll war. */
  readonly overload = signal(0);

  /** Vor dem Start setzen: PCM-Frames für den WAV-Export mitsammeln. */
  captureAudio = false;

  get hasAudio(): boolean {
    return this.captured.length > 0;
  }

  private ws: WebSocket | null = null;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private mediaStream: MediaStream | null = null;
  private handlers: StreamHandlers = {};
  private endResolve: (() => void) | null = null;
  private captured: Int16Array[] = [];

  async start(engine: Engine, modelId: string, handlers: StreamHandlers): Promise<void> {
    if (this.state() !== 'idle') {
      throw new Error('Es läuft bereits eine Sitzung. Bitte zuerst beenden.');
    }
    this.handlers = handlers;
    this.pendingConfig = { engine, modelId, sampleRate: TARGET_RATE };
    this.captured = [];
    this.overload.set(0);
    this.state.set('connecting');
    try {
      await this.openSocket();
      await this.openMicrophone();
      this.state.set('recording');
      handlers.onReady?.();
    } catch (error) {
      await this.teardown(false);
      this.state.set('idle');
      throw error instanceof Error ? error : new Error(String(error));
    }
  }

  async stop(): Promise<void> {
    if (this.state() === 'idle') {
      return;
    }
    this.state.set('stopping');
    this.stopMicrophone();
    const ws = this.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }));
      await new Promise<void>((resolve) => {
        this.endResolve = resolve;
        setTimeout(() => {
          if (this.endResolve) {
            this.endResolve = null;
            resolve();
          }
        }, 20_000);
      });
    }
    await this.teardown(true);
    this.state.set('idle');
  }

  abort(): void {
    this.endResolve = null;
    void this.teardown(true);
    this.state.set('idle');
  }

  // ---------------------------------------------------------------- sockets

  private openSocket(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${protocol}//${location.host}/ws/stream`);
      this.ws = ws;
      let settled = false;
      const timeout = setTimeout(() => {
        if (!settled) {
          settled = true;
          reject(new Error('Keine Verbindung zum Server (Timeout beim Modell-Laden).'));
          void this.teardown(false);
        }
      }, 180_000);

      ws.onopen = () => {
        const config = this.pendingConfig;
        ws.send(JSON.stringify({ type: 'config', ...config }));
      };
      ws.onmessage = (event) => {
        let message: { type?: string; text?: string; message?: string; start?: number; end?: number };
        try {
          message = JSON.parse(String(event.data));
        } catch {
          return;
        }
        switch (message.type) {
          case 'ready':
            if (!settled) {
              settled = true;
              clearTimeout(timeout);
              resolve();
            }
            break;
          case 'partial':
            this.handlers.onPartial?.(message.text ?? '');
            break;
          case 'final':
            if ((message.text ?? '').trim() || message.start !== undefined) {
              this.handlers.onFinal?.({
                text: message.text ?? '',
                start: message.start ?? 0,
                end: message.end ?? 0,
              });
            }
            break;
          case 'error':
            if (!settled) {
              settled = true;
              clearTimeout(timeout);
              reject(new Error(message.message || 'Stream-Fehler.'));
            } else {
              this.handlers.onError?.(message.message || 'Stream-Fehler.');
            }
            break;
          case 'end':
            if (!settled) {
              settled = true;
              clearTimeout(timeout);
              resolve();
            }
            this.handlers.onEnd?.();
            this.endResolve?.();
            this.endResolve = null;
            break;
        }
      };
      ws.onerror = () => {
        if (!settled) {
          settled = true;
          clearTimeout(timeout);
          reject(new Error('WebSocket-Fehler. Läuft der Server?'));
        }
      };
      ws.onclose = () => {
        if (!settled) {
          settled = true;
          clearTimeout(timeout);
          reject(new Error('Verbindung zum Server geschlossen.'));
        }
        this.endResolve?.();
        this.endResolve = null;
      };
    });
  }

  private pendingConfig: { engine?: Engine; modelId?: string; sampleRate?: number } = {};

  // -------------------------------------------------------------- microphone

  private async openMicrophone(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Dieser Browser unterstützt keine Mikrofonaufnahme (braucht HTTPS oder localhost).');
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.mediaStream = stream;
    const Context = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const context = new Context({ sampleRate: TARGET_RATE });
    this.audioContext = context;
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    this.source = source;
    this.processor = processor;
    const ratio = context.sampleRate / TARGET_RATE;
    processor.onaudioprocess = (event) => {
      if (this.state() !== 'recording' || this.ws?.readyState !== WebSocket.OPEN) {
        return;
      }
      const input = event.inputBuffer.getChannelData(0);
      const pcm = ratio === 1 ? floatToPcm16(input) : resampleToPcm16(input, ratio);
      if (pcm.byteLength === 0) {
        return;
      }
      if (this.captureAudio) {
        this.captured.push(pcm.slice());
      }
      // Wie in der Originial-HTML: kurzer Rückstau, Frames bei Überlast verwerfen.
      if (this.ws.bufferedAmount > 1_500_000) {
        this.overload.update((count) => count + 1);
        return;
      }
      this.ws.send(pcm.buffer);
    };
    source.connect(processor);
    processor.connect(context.destination);
  }

  private stopMicrophone(): void {
    this.processor?.disconnect();
    this.processor = null;
    this.source?.disconnect();
    this.source = null;
    void this.audioContext?.close();
    this.audioContext = null;
    this.mediaStream?.getTracks().forEach((track) => track.stop());
    this.mediaStream = null;
  }

  /** Gesammeltes Mikrofon-Audio als WAV (16-kHz-Mono), oder null. */
  takeAudio(): Blob | null {
    if (!this.captured.length) {
      return null;
    }
    const total = this.captured.reduce((sum, chunk) => sum + chunk.length, 0);
    const wav = new ArrayBuffer(44 + total * 2);
    const view = new DataView(wav);
    const write = (offset: number, text: string) => {
      for (let index = 0; index < text.length; index++) {
        view.setUint8(offset + index, text.charCodeAt(index));
      }
    };
    write(0, 'RIFF');
    view.setUint32(4, 36 + total * 2, true);
    write(8, 'WAVEfmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, TARGET_RATE, true);
    view.setUint32(28, TARGET_RATE * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    write(36, 'data');
    view.setUint32(40, total * 2, true);
    let offset = 44;
    for (const chunk of this.captured) {
      for (let index = 0; index < chunk.length; index++) {
        view.setInt16(offset, chunk[index], true);
        offset += 2;
      }
    }
    this.captured = [];
    return new Blob([wav], { type: 'audio/wav' });
  }

  clearAudio(): void {
    this.captured = [];
  }

  private async teardown(closeSocket: boolean): Promise<void> {
    this.stopMicrophone();
    if (closeSocket && this.ws && this.ws.readyState <= WebSocket.OPEN) {
      try {
        this.ws.close();
      } catch {
        /* egal */
      }
    }
    this.ws = null;
  }
}

function floatToPcm16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let index = 0; index < samples.length; index++) {
    const value = Math.max(-1, Math.min(1, samples[index]));
    out[index] = value < 0 ? value * 0x8000 : value * 0x7fff;
  }
  return out;
}

function resampleToPcm16(samples: Float32Array, ratio: number): Int16Array {
  const length = Math.max(1, Math.floor(samples.length / ratio));
  const out = new Int16Array(length);
  for (let index = 0; index < length; index++) {
    const position = index * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, samples.length - 1);
    const mix = position - left;
    const value = Math.max(-1, Math.min(1, samples[left] * (1 - mix) + samples[right] * mix));
    out[index] = value < 0 ? value * 0x8000 : value * 0x7fff;
  }
  return out;
}
