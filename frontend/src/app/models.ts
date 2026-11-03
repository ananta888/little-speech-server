export type Engine = 'vosk' | 'whisper';

export interface ModelEntry {
  id: string;
  engine: Engine | 'recasepunc';
  name: string;
  language: string | null;
  size: string | null;
  bytes: number | null;
  license: string | null;
  kind: string;
  available: boolean;
  downloading: boolean;
  downloaded: number | null;
  download_total: number | null;
  percent: number | null;
  error: string | null;
  note: string | null;
  url: string | null;
}

export interface ModelsResponse {
  vosk: ModelEntry[];
  whisper: ModelEntry[];
  recasepunc: ModelEntry;
  device: string;
  models_dir: string;
}

export interface HealthResponse {
  ok: boolean;
  device: string;
  device_requested: string;
  cuda: boolean;
  vosk_loaded: string | null;
  whisper_loaded: string | null;
  recasepunc_loaded: boolean;
  ffmpeg: boolean;
}

export interface DeviceResponse {
  requested: string;
  resolved: string;
  cuda_available: boolean;
  unloaded: string[];
}

export interface Word {
  word: string;
  start: number;
  end: number;
  conf: number | null;
}

export interface Segment {
  text: string;
  start: number;
  end: number;
}

export interface TranscribeResponse {
  engine: string;
  model: string;
  text: string;
  language: string | null;
  duration: number | null;
  words: Word[] | null;
  segments: Segment[];
  took_ms: number;
}

export interface PostprocessStatus {
  available: boolean;
  loaded: boolean;
  loading: boolean;
  device: string;
}

export const STATUS_TEXT: Record<string, string> = {
  downloading: 'wird heruntergeladen',
  done: 'fertig',
  error: 'Fehler',
  stopped: 'abgebrochen',
};
