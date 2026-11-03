import { Component, computed, OnDestroy, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from './api.service';
import { StreamService } from './stream.service';
import { Engine, ModelEntry, Segment, Word } from './models';

interface VoskGroup {
  name: string;
  options: ModelEntry[];
}

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit, OnDestroy {
  readonly engine = signal<Engine>('vosk');
  readonly voskModel = signal('');
  readonly whisperModel = signal('base');
  readonly status = signal('Modell wählen, dann transkribieren.');
  readonly error = signal('');
  readonly busy = signal<'idle' | 'upload' | 'post'>('idle');
  readonly transcript = signal('');
  readonly backup = signal<string | null>(null);
  readonly partial = signal('');
  readonly segments = signal<Segment[]>([]);
  readonly words = signal<Word[]>([]);
  readonly info = signal('');

  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private fileInput: HTMLInputElement | null = null;

  constructor(readonly api: ApiService, readonly stream: StreamService) {}

  readonly selectedFileSignal = signal<File | null>(null);

  readonly selected = computed<ModelEntry | null>(() => {
    const models = this.api.models();
    if (!models) {
      return null;
    }
    const list = this.engine() === 'vosk' ? models.vosk : models.whisper;
    const id = this.engine() === 'vosk' ? this.voskModel() : this.whisperModel();
    return list.find((entry) => entry.id === id) ?? null;
  });

  readonly groups = computed<VoskGroup[]>(() => {
    const models = this.api.models();
    const engine = this.engine();
    if (!models) {
      return [];
    }
    const list = engine === 'vosk' ? models.vosk : models.whisper;
    const order = (entry: ModelEntry): number => {
      if (engine === 'whisper') {
        return 0;
      }
      if (entry.language === 'Deutsch') {
        return 0;
      }
      if (entry.language === 'Englisch') {
        return 1;
      }
      if (entry.kind !== 'speech') {
        return 4;
      }
      if ((entry.name ?? '').startsWith('Weitere')) {
        return 3;
      }
      return 2;
    };
    const sorted = [...list].sort(
      (a, b) => order(a) - order(b) || (a.language ?? '').localeCompare(b.language ?? '', 'de'),
    );
    const groups: VoskGroup[] = [];
    for (const entry of sorted) {
      const name =
        engine === 'whisper'
          ? 'Whisper-Modelle'
          : entry.kind !== 'speech'
            ? 'Zusatzmodelle · keine Transkription'
            : entry.language ?? 'Weitere';
      let group = groups.find((candidate) => candidate.name === name);
      if (!group) {
        group = { name, options: [] };
        groups.push(group);
      }
      group.options.push(entry);
    }
    return groups;
  });

  readonly hasDownloading = computed(() => {
    const models = this.api.models();
    if (!models) {
      return false;
    }
    return [...models.vosk, ...models.whisper].some((entry) => entry.downloading);
  });

  readonly canWork = computed(() => {
    const model = this.selected();
    if (!model) {
      return false;
    }
    if (model.engine === 'vosk' && !model.available) {
      return false;
    }
    return this.busy() === 'idle' && this.stream.state() === 'idle';
  });

  readonly isRecording = computed(() => this.stream.state() !== 'idle');

  async ngOnInit(): Promise<void> {
    try {
      const models = await this.api.refreshModels();
      if (!this.voskModel()) {
        const preferred = models.vosk.find((entry) => entry.id === 'vosk-model-small-de-0.15');
        const german = models.vosk.find((entry) => entry.language === 'Deutsch' && entry.kind === 'speech');
        const firstAvailable = models.vosk.find((entry) => entry.available && entry.kind === 'speech');
        this.voskModel.set((preferred ?? firstAvailable ?? german ?? models.vosk[0])?.id ?? '');
      }
      this.status.set('Modell wählen; Modelle werden beim ersten Einsatz geladen.');
    } catch (error) {
      void this.fail(error, 'Modellliste konnte nicht geladen werden.');
    }
    try {
      await this.api.refreshHealth();
      await this.api.refreshPostprocess();
    } catch {
      /* Status ist optional */
    }
  }

  ngOnDestroy(): void {
    if (this.pollTimer) {
      clearTimeout(this.pollTimer);
    }
    this.stream.abort();
  }

  // ------------------------------------------------------------------ models

  async downloadSelected(): Promise<void> {
    const model = this.selected();
    if (!model) {
      return;
    }
    this.error.set('');
    try {
      if (model.engine === 'whisper') {
        await this.api.downloadWhisper(model.id);
      } else {
        await this.api.downloadVosk(model.id);
      }
      this.status.set(`${model.id}: Download gestartet …`);
      void this.api.refreshModels();
      this.schedulePoll();
    } catch (error) {
      void this.fail(error, 'Download konnte nicht gestartet werden.');
    }
  }

  async stopDownload(): Promise<void> {
    const model = this.selected();
    if (!model) {
      return;
    }
    const key = model.engine === 'whisper' ? `whisper:${model.id}` : model.id;
    try {
      await this.api.stopDownload(key);
      this.status.set(`${model.id}: Download abgebrochen.`);
      void this.api.refreshModels();
      this.schedulePoll();
    } catch (error) {
      void this.fail(error, 'Abbruch fehlgeschlagen.');
    }
  }

  private schedulePoll(): void {
    if (this.pollTimer) {
      return;
    }
    this.pollTimer = setTimeout(async () => {
      this.pollTimer = null;
      try {
        const models = await this.api.refreshModels();
        const stillDownloading = [...models.vosk, ...models.whisper].some((entry) => entry.downloading);
        if (stillDownloading) {
          this.schedulePoll();
        } else {
          const current = this.selected();
          if (current?.available) {
            this.status.set(`${current.id}: heruntergeladen und bereit.`);
          }
          await this.api.refreshHealth();
        }
      } catch {
        this.schedulePoll();
      }
    }, 1500);
  }

  // ------------------------------------------------------------------ upload

  onFileSelected(event: Event, input: HTMLInputElement): void {
    const file = (event.target as HTMLInputElement).files?.[0] ?? null;
    this.selectedFileSignal.set(file);
    this.fileInput = input;
    if (file) {
      this.status.set(`${file.name} gewählt. „Transkribieren“ startet die Erkennung.`);
      this.error.set('');
    }
  }

  async transcribeFile(): Promise<void> {
    const file = this.selectedFileSignal();
    const model = this.selected();
    if (!file || !model || this.busy() !== 'idle') {
      return;
    }
    this.busy.set('upload');
    this.error.set('');
    this.status.set(`${file.name} wird transkribiert …`);
    const previous = this.transcript();
    if (previous.trim()) {
      this.backup.set(previous);
    }
    try {
      const result = await this.api.transcribe(file, this.engine(), model.id);
      this.transcript.set(result.text);
      this.segments.set(result.segments);
      this.words.set(result.words ?? []);
      const details = [
        result.duration ? `${result.duration.toFixed(1)} s Audio` : null,
        `${result.took_ms} ms`,
        result.language ? `Sprache ${result.language}` : null,
      ].filter(Boolean);
      this.info.set(details.join(' · '));
      this.status.set(`Fertig: ${model.id}.`);
      if (this.fileInput) {
        this.fileInput.value = '';
      }
      this.selectedFileSignal.set(null);
    } catch (error) {
      void this.fail(error, 'Transkription fehlgeschlagen.');
    } finally {
      this.busy.set('idle');
    }
  }

  // ---------------------------------------------------------------- streaming

  async toggleRecording(): Promise<void> {
    if (this.isRecording()) {
      await this.stopRecording();
      return;
    }
    const model = this.selected();
    if (!model || this.busy() !== 'idle') {
      return;
    }
    this.error.set('');
    this.status.set(`Verbindung wird aufgebaut · ${model.id} wird geladen …`);
    try {
      if (this.transcript().trim()) {
        this.backup.set(this.transcript());
      }
      this.segments.set([]);
      this.words.set([]);
      this.info.set('');
      this.transcript.set('');
      await this.stream.start(this.engine(), model.id, {
        onReady: () => this.status.set('Aufnahme läuft. Sprechen; „Beenden“ schreibt das Ergebnis ins Transkript.'),
        onPartial: (text) => this.partial.set(text),
        onFinal: (segment) => this.appendLiveSegment(segment),
        onError: (message) => this.error.set(message),
        onEnd: () => this.partial.set(''),
      });
    } catch (error) {
      void this.fail(error, 'Live-Erkennung konnte nicht starten.');
    }
  }

  async stopRecording(): Promise<void> {
    this.status.set('Aufnahme wird beendet …');
    try {
      await this.stream.stop();
      this.partial.set('');
      this.status.set('Live-Erkennung beendet. Transkript kann bearbeitet werden.');
      const count = this.segments().length;
      this.info.set(count ? `${count} Abschnitte live erkannt` : '');
    } catch (error) {
      void this.fail(error, 'Beenden fehlgeschlagen.');
    }
  }

  private appendLiveSegment(segment: { text: string; start: number; end: number; words?: unknown[] }): void {
    const text = segment.text.trim();
    if (!text) {
      return;
    }
    const current = this.transcript().trim();
    this.transcript.set(current ? `${current} ${text}` : text);
    this.segments.update((list) => [...list, { text, start: segment.start, end: segment.end }]);
    if (Array.isArray(segment.words) && segment.words.length) {
      this.words.update((list) => [...list, ...(segment.words as Word[])]);
    }
    this.partial.set('');
  }

  // ------------------------------------------------------------- postprocess

  async recasepunc(): Promise<void> {
    const text = this.transcript();
    if (!text.trim() || this.busy() !== 'idle') {
      return;
    }
    this.busy.set('post');
    this.error.set('');
    this.backup.set(text);
    this.status.set('Zeichensetzung wird nachgezogen (recasepunc, lokal) …');
    try {
      const result = await this.api.postprocess(text);
      this.transcript.set(result);
      this.status.set('Zeichensetzung nachgezogen. Bitte prüfen; „Wiederherstellen“ zeigt das Original.');
      await this.api.refreshPostprocess();
    } catch (error) {
      void this.fail(error, 'Zeichensetzung fehlgeschlagen.');
    } finally {
      this.busy.set('idle');
    }
  }

  restore(): void {
    const backup = this.backup();
    if (backup !== null) {
      this.transcript.set(backup);
      this.backup.set(null);
      this.status.set('Original wiederhergestellt.');
    }
  }

  async downloadRecasepunc(): Promise<void> {
    this.error.set('');
    try {
      await this.api.downloadVosk(this.api.models()?.recasepunc.id ?? 'vosk-recasepunc-de-0.21');
      this.status.set('recasepunc-Download gestartet …');
      void this.api.refreshModels();
      this.schedulePoll();
    } catch (error) {
      void this.fail(error, 'Download konnte nicht gestartet werden.');
    }
  }

  // ------------------------------------------------------------------ export

  async copyOutput(): Promise<void> {
    try {
      await navigator.clipboard.writeText(this.transcript());
      this.status.set('Transkript kopiert.');
    } catch {
      this.status.set('Kopieren nicht möglich – Text markieren und Strg+C drücken.');
    }
  }

  saveTxt(): void {
    this.saveFile(this.transcript(), 'transkript.txt', 'text/plain;charset=utf-8');
  }

  saveSrt(): void {
    this.saveFile(this.buildSrt(), 'transkript.srt', 'application/x-subrip;charset=utf-8');
  }

  clearTranscript(): void {
    if (!this.transcript()) {
      return;
    }
    this.backup.set(this.transcript());
    this.transcript.set('');
    this.segments.set([]);
    this.words.set([]);
    this.info.set('');
    this.status.set('Transkript geleert (über „Wiederherstellen“ zurückholen).');
  }

  private buildSrt(): string {
    const entries = this.srtEntries();
    return entries
      .map(
        (entry, index) =>
          `${index + 1}\n${srtTime(entry.start)} --> ${srtTime(entry.end)}\n${entry.text}\n`,
      )
      .join('\n');
  }

  private srtEntries(): Segment[] {
    const segments = this.segments().filter((entry) => entry.text.trim());
    if (segments.length) {
      return segments;
    }
    const text = this.transcript().trim();
    if (!text) {
      return [];
    }
    const words = this.words();
    if (words.length > 4) {
      const lines: Segment[] = [];
      let current: Word[] = [];
      const flush = () => {
        if (current.length) {
          lines.push({
            text: current.map((word) => word.word).join(' '),
            start: current[0].start,
            end: current[current.length - 1].end,
          });
          current = [];
        }
      };
      for (const word of words) {
        current.push(word);
        const length = current.reduce((sum, entry) => sum + entry.word.length + 1, 0);
        if (length > 42) {
          flush();
        }
      }
      flush();
      return lines;
    }
    return [{ text, start: 0, end: Math.max(2, text.length / 15) }];
  }

  private saveFile(content: string, name: string, type: string): void {
    if (!content.trim()) {
      this.status.set('Kein Inhalt zum Speichern.');
      return;
    }
    const url = URL.createObjectURL(new Blob([content], { type }));
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    this.status.set(`${name} heruntergeladen.`);
  }

  // -------------------------------------------------------------------- misc

  isLowConfidence(word: Word): boolean {
    return word.conf !== null && word.conf < 0.7;
  }

  groupLabel(group: VoskGroup): string {
    if (group.name === 'Whisper-Modelle') {
      return group.name;
    }
    const available = group.options.filter((entry) => entry.available).length;
    return `${group.name} · ${available}/${group.options.length} geladen`;
  }

  optionLabel(entry: ModelEntry): string {
    const size = entry.size ? ` · ${entry.size}` : '';
    const mark = entry.available ? ' ✓' : entry.downloading ? ' · lädt …' : '';
    return `${entry.id.replace(/^vosk-model-/, '')}${size}${mark}`;
  }

  private async fail(error: unknown, prefix: string): Promise<void> {
    const message = await this.api.errorText(error);
    this.error.set(`${prefix} ${message}`);
    this.status.set('Fehler – Details siehe Hinweis.');
  }
}

function pad(value: number): string {
  return String(Math.floor(value)).padStart(2, '0');
}

function srtTime(seconds: number): string {
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const secs = Math.floor(safe % 60);
  const millis = Math.round((safe % 1) * 1000);
  return `${pad(hours)}:${pad(minutes)}:${pad(secs)},${String(millis).padStart(3, '0')}`;
}
