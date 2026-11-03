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

  private ws: WebSocket | null = null;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private mediaStream: MediaStream | null = null;
  private handlers: StreamHandlers = {};
  private endResolve: (() => void) | null = null;

  async start(engine: Engine, modelId: string, handlers: StreamHandlers): Promise<void> {
    if (this.state() !== 'idle') {
      throw new Error('Es läuft bereits eine Sitzung. Bitte zuerst beenden.');
    }
    this.handlers = handlers;
    this.pendingConfig = { engine, modelId, sampleRate: TARGET_RATE };
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
      if (pcm.byteLength > 0) {
        this.ws.send(pcm.buffer);
      }
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
