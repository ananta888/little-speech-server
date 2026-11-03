import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import {
  DeviceResponse,
  Engine,
  HealthResponse,
  ModelsResponse,
  PostprocessStatus,
  TranscribeResponse,
} from './models';

@Injectable({ providedIn: 'root' })
export class ApiService {
  readonly models = signal<ModelsResponse | null>(null);
  readonly health = signal<HealthResponse | null>(null);
  readonly postprocessStatus = signal<PostprocessStatus | null>(null);

  constructor(private readonly http: HttpClient) {}

  async refreshModels(): Promise<ModelsResponse> {
    const value = await firstValueFrom(this.http.get<ModelsResponse>('/api/models'));
    this.models.set(value);
    return value;
  }

  async refreshHealth(): Promise<HealthResponse> {
    const value = await firstValueFrom(this.http.get<HealthResponse>('/api/health'));
    this.health.set(value);
    return value;
  }

  async refreshPostprocess(): Promise<PostprocessStatus> {
    const value = await firstValueFrom(
      this.http.get<PostprocessStatus>('/api/postprocess/status'),
    );
    this.postprocessStatus.set(value);
    return value;
  }

  async setDevice(device: 'auto' | 'cpu' | 'cuda'): Promise<DeviceResponse> {
    const value = await firstValueFrom(
      this.http.post<DeviceResponse>('/api/device', { device }),
    );
    await this.refreshHealth();
    return value;
  }

  downloadVosk(modelId: string): Promise<unknown> {
    return firstValueFrom(this.http.post(`/api/downloads/vosk/${modelId}`, {}));
  }

  downloadWhisper(modelId: string): Promise<unknown> {
    return firstValueFrom(this.http.post(`/api/downloads/whisper/${modelId}`, {}));
  }

  stopDownload(modelId: string): Promise<unknown> {
    return firstValueFrom(this.http.post(`/api/downloads/stop/${modelId}`, {}));
  }

  deleteModel(engine: 'vosk' | 'whisper' | 'recasepunc', id: string): Promise<{ deleted: string }> {
    return firstValueFrom(
      this.http.post<{ deleted: string }>('/api/models/delete', { engine, id }),
    );
  }

  async transcribe(file: File, engine: Engine, modelId: string): Promise<TranscribeResponse> {
    const form = new FormData();
    form.append('file', file, file.name);
    form.append('engine', engine);
    form.append('model', modelId);
    return firstValueFrom(this.http.post<TranscribeResponse>('/api/transcribe', form));
  }

  async postprocess(text: string): Promise<string> {
    const result = await firstValueFrom(
      this.http.post<{ text: string }>('/api/postprocess', { text }),
    );
    return result.text;
  }

  async errorText(error: unknown): Promise<string> {
    if (error && typeof error === 'object' && 'error' in error) {
      const inner = (error as { error?: unknown }).error;
      if (inner && typeof inner === 'object' && 'detail' in inner) {
        return String((inner as { detail: unknown }).detail);
      }
    }
    return error instanceof Error ? error.message : String(error);
  }
}
