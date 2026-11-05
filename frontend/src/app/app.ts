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
  readonly device = signal<'auto' | 'cpu' | 'cuda'>('auto');
  readonly captureAudio = signal(false);
  readonly audioReady = signal(false);
  readonly autoPost = signal(localStorage.getItem('lss-auto-post') === '1');
  readonly selectionLength = signal(0);
  readonly importProgress = signal<number | null>(null);
  readonly importedModels = signal<ModelEntry[]>([]);
  // Sektion 04 – Textaufgaben
  readonly llmModel = signal('qwen3-0.6b');
  readonly llmTask = signal<'summary' | 'bullets' | 'tasks' | 'cleanup'>('summary');
  readonly llmOutput = signal('');
  readonly llmBusy = signal(false);
  readonly llmStatus = signal('Transkript aus03 verwenden, Aufgabe wählen und starten.');
  readonly llmError = signal('');
  private llmAbort: AbortController | null = null;
  private llmSections: string[] = [];
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
    const list = engine === 'vosk'
      ? [...models.vosk, ...this.importedModels().filter(
          (extra) => !models.vosk.some((entry) => entry.id === extra.id),
        )]
      : models.whisper;
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
    return this.busy() === 'idle' && this.stream.state() === 'idle' && !this.llmBusy();
  });

  readonly isRecording = computed(() => this.stream.state() !== 'idle');

  readonly gpuAvailable = computed(() => this.api.health()?.cuda === true);

  readonly activeDeviceLabel = computed(() => {
    const health = this.api.health();
    if (!health) {
      return '…';
    }
    return health.device === 'cuda' ? 'GPU (CUDA)' : 'CPU';
  });

  readonly llmEntry = computed<ModelEntry | null>(
    () => this.api.models()?.llm.find((entry) => entry.id === this.llmModel()) ?? null,
  );

  readonly llmLoadedId = computed(() => this.api.health()?.text_loaded ?? null);

  readonly llmInfo = computed(() => {
    const entry = this.llmEntry();
    if (!entry) {
      return '';
    }
    const parts = [`Download ${entry.size ?? '…'}`];
    if (entry.note) {
      parts.push(entry.note);
    }
    parts.push(this.llmLoadedId() === entry.id ? 'geladen' : entry.available ? 'bereit' : 'nicht geladen');
    return parts.join(' · ');
  });

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
      const health = await this.api.refreshHealth();
      this.device.set(
        health.device_requested === 'cpu' || health.device_requested === 'cuda'
          ? health.device_requested
          : 'auto',
      );
      await this.api.refreshPostprocess();
    } catch {
      /* Status ist optional */
    }
  }

  async changeDevice(value: 'auto' | 'cpu' | 'cuda'): Promise<void> {
    if (this.busy() !== 'idle' || this.isRecording()) {
      return;
    }
    this.device.set(value);
    this.error.set('');
    try {
      const result = await this.api.setDevice(value);
      await this.api.refreshModels();
      const label = result.resolved === 'cuda' ? 'GPU (CUDA)' : 'CPU';
      let message = `Ausführung: ${result.requested} → aktiv: ${label}.`;
      if (value === 'cuda' && result.resolved !== 'cuda') {
        message += ' CUDA ist auf dem Server nicht verfügbar – es bleibt bei der CPU.';
      }
      if (result.unloaded.length) {
        message += ` ${result.unloaded.join(' und ')} entladen; geladene Modelle werden beim nächsten Einsatz neu geladen.`;
      }
      this.status.set(message);
    } catch (error) {
      void this.fail(error, 'Gerät konnte nicht gewechselt werden.');
      const requested = this.api.health()?.device_requested ?? 'auto';
      this.device.set(requested === 'cpu' || requested === 'cuda' ? requested : 'auto');
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
    let succeeded = false;
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
      succeeded = true;
      if (this.fileInput) {
        this.fileInput.value = '';
      }
      this.selectedFileSignal.set(null);
    } catch (error) {
      void this.fail(error, 'Transkription fehlgeschlagen.');
    } finally {
      this.busy.set('idle');
    }
    if (succeeded) {
      await this.autoPostprocess();
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
      this.audioReady.set(false);
      this.stream.captureAudio = this.captureAudio();
      this.stream.clearAudio();
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
      this.audioReady.set(this.stream.hasAudio);
      this.status.set('Live-Erkennung beendet. Transkript kann bearbeitet werden.');
      const count = this.segments().length;
      this.info.set(count ? `${count} Abschnitte live erkannt` : '');
      await this.autoPostprocess();
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
      const result = await this.postprocessChunked(text, 'Zeichensetzung');
      this.transcript.set(result);
      this.status.set('Zeichensetzung nachgezogen. Bitte prüfen; „Wiederherstellen“ zeigt das Original.');
      await this.api.refreshPostprocess();
    } catch (error) {
      void this.fail(error, 'Zeichensetzung fehlgeschlagen.');
    } finally {
      this.busy.set('idle');
    }
  }

  async postprocessSelection(): Promise<void> {
    const element = document.getElementById('transcript') as HTMLTextAreaElement | null;
    const start = element?.selectionStart ?? 0;
    const end = element?.selectionEnd ?? 0;
    const text = this.transcript();
    if (!element || start === end || start >= end || this.busy() !== 'idle') {
      this.status.set('Bitte zuerst im Transkript einen Textbereich markieren.');
      return;
    }
    const selection = text.slice(start, end);
    if (!selection.trim()) {
      this.status.set('Die Auswahl enthält keinen Text.');
      return;
    }
    this.busy.set('post');
    this.error.set('');
    this.backup.set(text);
    this.status.set(`Auswahl wird nachbearbeitet (${selection.length} Zeichen) …`);
    try {
      const replaced = await this.postprocessChunked(selection, 'Auswahl');
      const next = text.slice(0, start) + replaced + text.slice(end);
      this.transcript.set(next);
      this.selectionLength.set(replaced.length);
      this.status.set('Auswahl nachbearbeitet. „Wiederherstellen“ zeigt das Original.');
      await this.api.refreshPostprocess();
      requestAnimationFrame(() => {
        const area = document.getElementById('transcript') as HTMLTextAreaElement | null;
        if (area) {
          area.focus();
          area.setSelectionRange(start, start + replaced.length);
        }
      });
    } catch (error) {
      void this.fail(error, 'Nachbearbeitung der Auswahl fehlgeschlagen.');
    } finally {
      this.busy.set('idle');
    }
  }

  updateSelection(event: Event): void {
    const element = event.target as HTMLTextAreaElement;
    this.selectionLength.set(Math.max(0, element.selectionEnd - element.selectionStart));
  }

  setAutoPost(value: boolean): void {
    this.autoPost.set(value);
    localStorage.setItem('lss-auto-post', value ? '1' : '0');
  }

  /** Auto-Modus: nach Transkriptionsende direkt nachziehen (wenn Modell vorhanden). */
  private async autoPostprocess(): Promise<void> {
    if (!this.autoPost() || this.busy() !== 'idle' || !this.transcript().trim()) {
      return;
    }
    await this.api.refreshPostprocess();
    const status = this.api.postprocessStatus();
    if (!status?.available) {
      this.status.set('Auto-Nachbearbeitung übersprungen: recasepunc-Modell ist nicht geladen (Abschnitt 03).');
      return;
    }
    const raw = this.transcript();
    this.busy.set('post');
    this.backup.set(raw);
    this.status.set('Auto-Nachbearbeitung: Zeichensetzung wird nachgezogen …');
    try {
      this.transcript.set(await this.postprocessChunked(raw, 'Auto-Nachbearbeitung'));
      this.status.set('Auto-Nachbearbeitung fertig. „Wiederherstellen“ zeigt das rohe Transkript.');
    } catch (error) {
      void this.fail(error, 'Auto-Nachbearbeitung fehlgeschlagen.');
    } finally {
      this.busy.set('idle');
    }
  }

  /** Long Texte abschnittsweise, damit Fortschritt und Fehler pro Abschnitt möglich sind. */
  private async postprocessChunked(text: string, label: string): Promise<string> {
    const chunks: string[] = [];
    let current: string[] = [];
    let size = 0;
    for (const line of text.split('\n')) {
      if (size + line.length + 1 > 2000 && current.length) {
        chunks.push(current.join('\n'));
        current = [];
        size = 0;
      }
      current.push(line);
      size += line.length + 1;
    }
    if (current.length) {
      chunks.push(current.join('\n'));
    }
    if (!chunks.length) {
      return text;
    }
    const results: string[] = [];
    for (let index = 0; index < chunks.length; index++) {
      this.status.set(`${label} · Abschnitt ${index + 1} / ${chunks.length} …`);
      results.push(await this.api.postprocess(chunks[index]));
    }
    return results.join('\n');
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

  // ------------------------------------------------------- Sektion 04 · Text

  async downloadLlm(): Promise<void> {
    const entry = this.llmEntry();
    if (!entry) {
      return;
    }
    this.llmError.set('');
    try {
      await this.api.downloadLlm(entry.id);
      this.llmStatus.set(`${entry.name}: Download gestartet …`);
      void this.api.refreshModels();
      this.schedulePoll();
    } catch (error) {
      const message = await this.api.errorText(error);
      this.llmError.set(`Download konnte nicht gestartet werden. ${message}`);
    }
  }

  async stopDownloadLlm(): Promise<void> {
    const entry = this.llmEntry();
    if (!entry) {
      return;
    }
    try {
      await this.api.stopDownload(`llm:${entry.id}`);
      this.llmStatus.set(`${entry.name}: Download abgebrochen.`);
      void this.api.refreshModels();
      this.schedulePoll();
    } catch (error) {
      const message = await this.api.errorText(error);
      this.llmError.set(message);
    }
  }

  async unloadTextModel(): Promise<void> {
    try {
      const result = await this.api.unloadText();
      await this.api.refreshHealth();
      this.llmStatus.set(result.unloaded ? 'Textmodell entladen.' : 'Kein Textmodell geladen.');
    } catch (error) {
      const message = await this.api.errorText(error);
      this.llmError.set(message);
    }
  }

  async runTextTask(): Promise<void> {
    const text = this.transcript();
    if (!text.trim() || this.llmBusy() || this.busy() !== 'idle' || this.isRecording()) {
      return;
    }
    const entry = this.llmEntry();
    if (entry && entry.note?.includes('GPU') && !this.gpuAvailable()) {
      this.llmError.set('Dieses Textmodell benötigt eine GPU – Gerät in Abschnitt 01 umstellen.');
      return;
    }
    this.llmBusy.set(true);
    this.llmError.set('');
    this.llmOutput.set('');
    this.llmSections = [];
    this.llmStatus.set('Verbindung wird aufgebaut …');
    this.llmAbort = new AbortController();
    try {
      const response = await fetch('/api/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, task: this.llmTask(), model: this.llmModel() }),
        signal: this.llmAbort.signal,
      });
      if (!response.ok || !response.body) {
        let detail = `HTTP ${response.status}`;
        try {
          const body = await response.json();
          if (typeof body.detail === 'string') {
            detail = body.detail;
          }
        } catch {
          /* ohne JSON-Antwort */
        }
        throw new Error(detail);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        let newline = buffer.indexOf('\n');
        while (newline >= 0) {
          const line = buffer.slice(0, newline);
          buffer = buffer.slice(newline + 1);
          if (line.trim()) {
            this.handleTextEvent(JSON.parse(line));
          }
          newline = buffer.indexOf('\n');
        }
      }
      await this.api.refreshHealth();
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        this.llmStatus.set('Abgebrochen. Ein sichtbarer Entwurf bleibt erhalten.');
      } else {
        const message = await this.api.errorText(error);
        this.llmError.set(message);
        this.llmStatus.set('Verarbeitung nicht abgeschlossen.');
      }
    } finally {
      this.llmAbort = null;
      this.llmBusy.set(false);
      void this.api.refreshModels();
    }
  }

  private handleTextEvent(event: {
    type: string;
    status?: string;
    file?: string;
    loaded?: number;
    total?: number;
    model?: string;
    device?: string;
    index?: number;
    merging?: boolean;
    label?: string;
    text?: string;
    sections?: number;
    took_ms?: number;
    message?: string;
  }): void {
    switch (event.type) {
      case 'download':
        if (event.status === 'progress' && event.total) {
          this.llmStatus.set(
            `Modell laden: ${event.file ?? ''} · ${((event.loaded ?? 0) / 1e6).toFixed(1)} / ${((event.total) / 1e6).toFixed(1)} MB`,
          );
        } else if (event.status === 'done') {
          this.llmStatus.set('Modelldateien vorbereiten …');
        }
        break;
      case 'load':
        this.llmStatus.set(`Modell wird geladen (${event.device === 'cuda' ? 'GPU' : 'CPU'}) …`);
        break;
      case 'preview':
        this.llmOutput.set(this.sectionDraft(event.text ?? ''));
        this.llmStatus.set(
          `${event.label ?? 'Text verarbeiten'} · Abschnitt ${event.index} / ${event.total} · Entwurf`,
        );
        break;
      case 'section':
        if (event.index && event.text !== undefined) {
          this.llmSections[event.index - 1] = event.text;
        }
        this.llmOutput.set(this.sectionDraft(''));
        this.llmStatus.set(`Abschnitt ${event.index} / ${event.total} fertig`);
        break;
      case 'done':
        this.llmOutput.set(event.text ?? '');
        this.llmStatus.set(
          `Fertig · ${event.sections ?? 1} Abschnitt${(event.sections ?? 1) === 1 ? '' : 'e'} · ${event.took_ms ?? 0} ms. Ergebnis prüfen und bei Bedarf speichern.`,
        );
        break;
      case 'cancelled':
        if (event.text) {
          this.llmOutput.set(event.text);
        }
        this.llmStatus.set('Abgebrochen. Der bis dahin erzeugte Teil bleibt als Entwurf erhalten.');
        break;
      case 'error':
        this.llmError.set(event.message || 'Unbekannter Fehler im Textdienst.');
        this.llmStatus.set('Verarbeitung fehlgeschlagen.');
        break;
    }
  }

  private sectionDraft(preview: string): string {
    const done = this.llmSections.filter((section) => section !== undefined && section !== '').join('\n\n');
    if (!preview) {
      return done;
    }
    return done ? `${done}\n\n${preview}` : preview;
  }

  cancelTextTask(): void {
    void this.api.cancelText();
    this.llmAbort?.abort();
  }

  async copyLlmOutput(): Promise<void> {
    try {
      await navigator.clipboard.writeText(this.llmOutput());
      this.llmStatus.set('Ergebnis kopiert.');
    } catch {
      this.llmStatus.set('Kopieren nicht möglich – Text markieren und Strg+C drücken.');
    }
  }

  saveLlmTxt(): void {
    this.saveFile(this.llmOutput(), 'textergebnis.txt', 'text/plain;charset=utf-8');
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

  saveJson(): void {
    if (!this.transcript().trim()) {
      this.status.set('Kein Inhalt zum Speichern.');
      return;
    }
    const model = this.selected();
    const payload = {
      created: new Date().toISOString(),
      engine: this.engine(),
      model: model?.id ?? null,
      info: this.info(),
      text: this.transcript(),
      segments: this.segments(),
      words: this.words(),
    };
    this.saveFile(JSON.stringify(payload, null, 2), 'transkript.json', 'application/json');
  }

  saveAudio(): void {
    const blob = this.stream.takeAudio();
    if (!blob) {
      this.audioReady.set(false);
      this.status.set('Kein aufgezeichnetes Audio vorhanden.');
      return;
    }
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `aufnahme-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.wav`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    this.audioReady.set(false);
    this.status.set('Aufnahme als WAV heruntergeladen (16 kHz, mono).');
  }

  async importArchive(event: Event, input: HTMLInputElement): Promise<void> {
    const file = (event.target as HTMLInputElement).files?.[0] ?? null;
    if (!file || this.busy() !== 'idle' || this.isRecording()) {
      return;
    }
    this.error.set('');
    this.engine.set('vosk');
    this.status.set(`${file.name} wird auf den Server geladen …`);
    this.importProgress.set(0);
    try {
      const result = await this.api.importArchive(file, (loaded, total) => {
        const percent = Math.round((loaded / total) * 100);
        this.importProgress.set(percent);
        this.status.set(`${file.name} wird auf den Server geladen … ${percent} % (${(loaded / 1e6).toFixed(0)} MB)`);
      });
      const entry: ModelEntry = {
        id: result.id,
        engine: 'vosk',
        name: `Importiert · ${result.id}`,
        language: 'Importiert',
        size: `${(file.size / 1e6).toFixed(1)} MB`,
        bytes: file.size,
        license: 'unbekannt (eigenes Archiv)',
        kind: 'speech',
        available: true,
        downloading: false,
        downloaded: null,
        download_total: null,
        percent: null,
        error: null,
        note: 'Lokal importiertes Archiv.',
        url: null,
      };
      this.importedModels.update((list) => [...list.filter((m) => m.id !== entry.id), entry]);
      this.voskModel.set(result.id);
      await this.api.refreshModels();
      this.status.set(`Importiert: ${result.id}. Modell ist wählbar und bereit.`);
    } catch (error) {
      void this.fail(error, 'Import fehlgeschlagen.');
    } finally {
      this.importProgress.set(null);
      input.value = '';
    }
  }

  async deleteModel(): Promise<void> {
    const model = this.selected();
    if (!model || this.busy() !== 'idle') {
      return;
    }
    if (!window.confirm(`Modell ${model.id} vom Server löschen? Die Dateien müssen bei Bedarf neu heruntergeladen werden.`)) {
      return;
    }
    this.error.set('');
    try {
      await this.api.deleteModel(model.engine as 'vosk' | 'whisper' | 'recasepunc', model.id);
      this.importedModels.update((list) => list.filter((entry) => entry.id !== model.id));
      await this.api.refreshModels();
      await this.api.refreshHealth();
      await this.api.refreshPostprocess();
      this.status.set(`Gelöscht: ${model.id}.`);
    } catch (error) {
      void this.fail(error, 'Löschen fehlgeschlagen.');
    }
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
