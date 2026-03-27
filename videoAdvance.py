#!/usr/bin/env python3
"""
Scene Improvement Engine  (v2 — Subtitle-Aware)
================================================
Analyzes scenes through two legendary lenses:
  🎬 Christopher Nolan — Temporal complexity, psychological depth, moral ambiguity
  🎭 William Shakespeare — Poetic meter, dramatic irony, soliloquy, tragedy/comedy

SUBTITLE SOURCES (tried in this priority order):
  1. External .srt / .vtt / .ass file alongside the video
  2. Embedded subtitle track inside the video (MKV, MP4, etc.) via ffmpeg
  3. Whisper speech-to-text (audio → timed segments, auto-fallback)
  4. --text / --transcript flags for plain text input

Subtitles give us:
  • Speaker turns and identity (who speaks, who goes silent)
  • Precise timestamps for scene segmentation (silence gap = scene break)
  • Dialogue density, pace, and rhythm per scene
  • Silence as subtext — pauses > 2s are dramatically significant
  • Repeated words as motifs / obsession signals

Dependencies:
    pip install openai-whisper moviepy rich requests
    ffmpeg must be on PATH (https://ffmpeg.org) for embedded subtitle extraction

Optional LLM (best quality, fully local):
    ollama pull llama3.1:8b   (from https://ollama.com)
"""

import os
import sys
import json
import re
import subprocess
import argparse
import tempfile
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    RICH = True
except ImportError:
    RICH = False
    print("Install rich for best output: pip install rich")

console = Console() if RICH else None


def log(msg, style=""):
    if console:
        console.print(msg, style=style)
    else:
        print(re.sub(r'\[/?[a-zA-Z0-9_ ]*\]', '', msg))


# ─────────────────────────────────────────────────────────────────────────────
#  SUBTITLE DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class SubtitleLine:
    def __init__(self, index, start, end, text, speaker=""):
        self.index = index
        self.start = start        # seconds (float)
        self.end = end            # seconds (float)
        self.text = text.strip()
        self.speaker = speaker    # populated if SRT uses "SPEAKER: text" format
        self.gap_before = 0.0    # silence gap before this cue (seconds)

    def __repr__(self):
        spk = f"{self.speaker}: " if self.speaker else ""
        return f"[{fmt_time(self.start)} → {fmt_time(self.end)}] {spk}{self.text}"


def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def srt_time_to_sec(ts):
    ts = ts.replace(",", ".").strip()
    try:
        h, m, s = ts.split(":")
        return float(h)*3600 + float(m)*60 + float(s)
    except Exception:
        return 0.0


def vtt_time_to_sec(ts):
    ts = ts.strip()
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        return float(parts[0])*60 + float(parts[1])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  SUBTITLE PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_srt(content):
    lines = []
    for block in re.split(r'\n\s*\n', content.strip()):
        parts = block.strip().splitlines()
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0].strip())
        except ValueError:
            continue
        tm = re.match(r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', parts[1])
        if not tm:
            continue
        start, end = srt_time_to_sec(tm.group(1)), srt_time_to_sec(tm.group(2))
        raw = " ".join(parts[2:]).strip()
        raw = re.sub(r'<[^>]+>', '', raw)           # strip HTML tags
        speaker = ""
        sm = re.match(r'^([A-Z][A-Z ]{1,20}):\s+(.+)', raw)
        if sm:
            speaker, raw = sm.group(1).title(), sm.group(2)
        lines.append(SubtitleLine(idx, start, end, raw, speaker))
    return lines


def parse_vtt(content):
    lines = []
    content = re.sub(r'^WEBVTT.*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'NOTE\s.*?\n\n', '', content, flags=re.DOTALL)
    idx = 0
    for block in re.split(r'\n\s*\n', content.strip()):
        parts = block.strip().splitlines()
        if not parts:
            continue
        time_line, text_start = parts[0], 1
        if '-->' not in time_line and len(parts) > 1:
            time_line, text_start = parts[1], 2
        tm = re.match(r'([\d:\.]+)\s*-->\s*([\d:\.]+)', time_line)
        if not tm:
            continue
        start, end = vtt_time_to_sec(tm.group(1)), vtt_time_to_sec(tm.group(2))
        raw = re.sub(r'<[^>]+>', '', " ".join(parts[text_start:]).strip())
        idx += 1
        lines.append(SubtitleLine(idx, start, end, raw))
    return lines


def parse_ass(content):
    lines = []
    idx = 0
    def ass_t(t):
        try:
            h, m, s = t.strip().split(":")
            return int(h)*3600 + int(m)*60 + float(s)
        except Exception:
            return 0.0
    for line in content.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        start, end = ass_t(parts[1]), ass_t(parts[2])
        raw = re.sub(r'\{[^}]*\}', '', parts[9]).replace("\\N", " ").replace("\\n", " ").strip()
        idx += 1
        lines.append(SubtitleLine(idx, start, end, raw))
    return lines


def load_subtitle_file(path):
    ext = Path(path).suffix.lower()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        log(f"[red]Cannot read subtitle file: {e}[/red]" if RICH else str(e))
        return []
    parsers = {".srt": parse_srt, ".vtt": parse_vtt, ".ass": parse_ass, ".ssa": parse_ass}
    subs = parsers.get(ext, parse_srt)(content)
    if not subs:
        log("[yellow]No subtitle cues found in file.[/yellow]" if RICH else "No cues found.")
        return []
    for i in range(1, len(subs)):
        subs[i].gap_before = max(0.0, subs[i].start - subs[i-1].end)
    log(f"[green]✓ Loaded {len(subs)} subtitle cues from {Path(path).name}[/green]" if RICH
        else f"✓ {len(subs)} cues loaded.")
    return subs


# ─────────────────────────────────────────────────────────────────────────────
#  SUBTITLE SOURCE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def find_external_subtitle(video_path):
    base = Path(video_path).with_suffix("")
    for ext in [".srt", ".en.srt", ".eng.srt", ".vtt", ".en.vtt", ".ass", ".ssa"]:
        candidate = Path(str(base) + ext)
        if candidate.exists():
            log(f"[green]✓ External subtitle found:[/green] {candidate.name}" if RICH
                else f"✓ External subtitle: {candidate.name}")
            return str(candidate)
    return None


def extract_embedded_subtitles(video_path, output_dir):
    """Extract first text-based subtitle track from video using ffmpeg."""
    out_srt = os.path.join(output_dir, "embedded_subs.srt")
    try:
        # Probe for subtitle streams
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "s", video_path],
            capture_output=True, text=True, timeout=30
        )
        if probe.returncode != 0:
            return None
        streams = json.loads(probe.stdout).get("streams", [])
        if not streams:
            log("[dim]No embedded subtitle tracks in this file.[/dim]" if RICH
                else "No embedded subtitles.")
            return None

        # Report what tracks exist
        log(f"[cyan]Found {len(streams)} embedded subtitle track(s):[/cyan]" if RICH
            else f"Found {len(streams)} subtitle track(s):")
        for s in streams:
            lang = s.get("tags", {}).get("language", "?")
            title = s.get("tags", {}).get("title", "")
            codec = s.get("codec_name", "?")
            log(f"  [dim]• Stream {s.get('index','?')} — codec: {codec}  lang: {lang}  {title}[/dim]"
                if RICH else f"  • Stream {s.get('index','?')} {codec} ({lang}) {title}")

        # Extract first subtitle stream as SRT
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", "-c:s", "srt", out_srt],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and os.path.exists(out_srt) and os.path.getsize(out_srt) > 20:
            log(f"[green]✓ Embedded subtitle extracted.[/green]" if RICH else "✓ Embedded subtitle extracted.")
            return out_srt
        else:
            log("[yellow]Embedded subtitle extraction failed (may be image-based: PGS/VOBSUB — not extractable as text).[/yellow]"
                if RICH else "Embedded subtitle extraction failed.")
            return None
    except FileNotFoundError:
        log("[yellow]ffmpeg not found. Install from https://ffmpeg.org for embedded subtitle support.[/yellow]"
            if RICH else "ffmpeg not found.")
        return None
    except Exception as e:
        log(f"[yellow]Embedded subtitle error: {e}[/yellow]" if RICH else str(e))
        return None


def whisper_timed_segments(video_path, model_size, output_dir):
    """Whisper fallback — returns timed SubtitleLine segments from audio."""
    log(f"[bold cyan]Transcribing with Whisper ({model_size}) — building timed segments...[/bold cyan]"
        if RICH else f"Whisper transcription ({model_size})...")
    try:
        import whisper
        from moviepy.editor import VideoFileClip
    except ImportError as e:
        log(f"[red]Missing: {e}  →  pip install openai-whisper moviepy[/red]" if RICH else str(e))
        sys.exit(1)

    audio_path = os.path.join(output_dir, "audio.wav")
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(audio_path, logger=None)

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, verbose=False, word_timestamps=True)

    subs = []
    for i, seg in enumerate(result.get("segments", [])):
        text = seg.get("text", "").strip()
        if not text:
            continue
        subs.append(SubtitleLine(i+1, seg.get("start", 0), seg.get("end", 0), text))

    for i in range(1, len(subs)):
        subs[i].gap_before = max(0.0, subs[i].start - subs[i-1].end)

    log(f"[green]✓ Whisper produced {len(subs)} timed segments.[/green]" if RICH
        else f"✓ {len(subs)} Whisper segments.")
    return subs


# ─────────────────────────────────────────────────────────────────────────────
#  SUBTITLE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_subtitle_structure(subs):
    """Extract rich dramatic metrics from subtitle timing and content."""
    if not subs:
        return {}
    duration = subs[-1].end - subs[0].start
    total_words = sum(len(s.text.split()) for s in subs)
    wpm = (total_words / duration * 60) if duration > 0 else 0

    # Silences > 2s are dramatically significant (subtext, hesitation, dread)
    big_silences = [(round(s.gap_before, 1), s.text[:60]) for s in subs if s.gap_before >= 2.0]
    avg_gap = sum(s.gap_before for s in subs) / len(subs)

    # Speaker word counts
    speakers = {}
    for s in subs:
        spk = s.speaker or "—"
        speakers[spk] = speakers.get(spk, 0) + len(s.text.split())

    # Emotional register
    questions = sum(1 for s in subs if "?" in s.text)
    exclamations = sum(1 for s in subs if "!" in s.text)

    # Motif detection — repeated words (possible obsessions)
    all_words = " ".join(s.text for s in subs).lower().split()
    freq = {}
    for w in all_words:
        if len(w) > 4:
            freq[w] = freq.get(w, 0) + 1
    top_words = sorted(freq.items(), key=lambda x: -x[1])[:10]

    lengths = [len(s.text.split()) for s in subs]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    return {
        "total_cues": len(subs),
        "total_words": total_words,
        "duration_seconds": round(duration, 1),
        "words_per_minute": round(wpm, 1),
        "average_gap_seconds": round(avg_gap, 2),
        "significant_silences": big_silences[:5],
        "speakers": speakers,
        "avg_line_words": round(avg_len, 1),
        "max_line_words": max(lengths) if lengths else 0,
        "question_lines": questions,
        "exclamation_lines": exclamations,
        "top_repeated_words": top_words,
        "pace": (
            "rapid-fire" if wpm > 160 else
            "fast" if wpm > 130 else
            "natural" if wpm > 100 else
            "deliberate" if wpm > 70 else
            "slow and weighty"
        )
    }


def segment_by_subtitles(subs, scene_gap=5.0):
    """Split subtitle cues into scenes using silence gaps as scene boundaries."""
    if not subs:
        return []
    scenes, current = [], [subs[0]]
    for sub in subs[1:]:
        if sub.gap_before >= scene_gap:
            scenes.append(current)
            current = [sub]
        else:
            current.append(sub)
    if current:
        scenes.append(current)

    result = []
    for sc in scenes:
        parts = [f"{s.speaker}: {s.text}" if s.speaker else s.text for s in sc]
        result.append({
            "subs": sc,
            "text": " ".join(parts),
            "start": sc[0].start,
            "end": sc[-1].end,
            "duration": sc[-1].end - sc[0].start,
            "line_count": len(sc),
        })
    return result


def build_subtitle_context(stats, scene_info):
    """Format subtitle analytics as readable LLM context."""
    if not stats:
        return "No subtitle timing data available."
    lines = [
        f"Dialogue pace: {stats.get('pace','')} ({stats.get('words_per_minute',0)} words/min)",
        f"Scene lines: {scene_info.get('line_count','?')}  |  Duration: {round(scene_info.get('duration',0),1)}s",
        f"Average silence between lines: {stats.get('average_gap_seconds',0)}s",
    ]
    silences = stats.get("significant_silences", [])
    if silences:
        lines.append("Dramatically significant silences (>2s):")
        for gap, text in silences[:3]:
            lines.append(f"  • {gap}s of silence before: \"{text}\"")
    speakers = stats.get("speakers", {})
    if len(speakers) > 1:
        lines.append("Speaker word counts: " + ", ".join(
            f"{k}={v}" for k,v in sorted(speakers.items(), key=lambda x: -x[1])))
    top = stats.get("top_repeated_words", [])
    if top:
        lines.append("Most repeated words (motifs / obsessions): " +
                     ", ".join(f"{w}({c})" for w,c in top[:6]))
    q, ex = stats.get("question_lines",0), stats.get("exclamation_lines",0)
    if q or ex:
        lines.append(f"Emotional register: {q} questions, {ex} exclamations")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  NLP METRICS (local, no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

NOLAN_METRICS = {
    "temporal_complexity": {
        "desc": "Does time bend, loop, fragment, or haunt the scene?",
        "signals": ["flashback","memory","future","parallel","non-linear","simultaneous","before","after","then","now"],
        "weight": 1.4
    },
    "psychological_depth": {
        "desc": "Are inner worlds — obsession, grief, doubt — made visible?",
        "signals": ["feel","think","believe","fear","obsess","remember","dream","real","truth","lie","mind","conscious"],
        "weight": 1.5
    },
    "moral_ambiguity": {
        "desc": "Is right and wrong genuinely unclear?",
        "signals": ["choice","sacrifice","cost","wrong","right","must","should","deserve","justify","evil","good"],
        "weight": 1.3
    },
    "structural_mirroring": {
        "desc": "Does the scene's structure echo its theme?",
        "signals": ["return","again","same","mirror","repeat","cycle","begin","end","always","never"],
        "weight": 1.2
    },
    "visual_subtext": {
        "desc": "Are emotions expressed through objects, space, silence?",
        "signals": ["silence","look","door","window","hand","eye","dark","light","still","watch","turn","away"],
        "weight": 1.3
    },
    "philosophical_tension": {
        "desc": "Does the scene wrestle with meaning, identity, or reality?",
        "signals": ["real","exist","meaning","purpose","who","what is","identity","matter","believe","question"],
        "weight": 1.2
    }
}

SHAKESPEARE_METRICS = {
    "iambic_rhythm": {"desc": "Does dialogue carry a natural da-DUM heartbeat?", "signals": [], "weight": 1.3},
    "dramatic_irony": {"desc": "Does the audience know something the character doesn't?",
        "signals": ["but he","but she","yet they","though I","if only","unaware","unseen","unknown"], "weight": 1.5},
    "soliloquy_potential": {"desc": "A private truth begging to be spoken aloud?",
        "signals": ["alone","myself","think","wonder","what if","perhaps","I must","cannot","dare I","should I"], "weight": 1.4},
    "conflict_polarity": {"desc": "Love/duty, life/death in direct collision?",
        "signals": ["love","duty","honor","death","life","war","peace","friend","enemy","heart","sword"], "weight": 1.4},
    "poetic_density": {"desc": "Richness of metaphor, imagery, figurative language",
        "signals": ["like","as","is a","becomes","burns","blooms","bleeds","rises","falls","sings","cries"], "weight": 1.2},
    "tragic_flaw_visibility": {"desc": "Is the character's hamartia on display?",
        "signals": ["pride","cannot","refuse","blind","stubborn","jealous","ambition","greed","anger"], "weight": 1.5},
    "catharsis_arc": {"desc": "Does the scene build toward emotional release?",
        "signals": ["realize","understand","finally","at last","now I see","forgive","release","truth","clear"], "weight": 1.3}
}


def count_syllables(word):
    word = word.lower().strip(".,!?;:'\"")
    count, prev_vowel = 0, False
    for c in word:
        iv = c in "aeiouy"
        if iv and not prev_vowel:
            count += 1
        prev_vowel = iv
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def measure_iambic(text):
    words = text.split()
    if len(words) < 4:
        return 0.5
    hits, total = 0, 0
    for i in range(0, len(words)-1, 2):
        if count_syllables(words[i]) <= count_syllables(words[i+1]):
            hits += 1
        total += 1
    return hits / total if total else 0.5


def detect_metaphors(text):
    found = []
    for pat in [r'\b\w+ (?:is|are|was|were) (?:a|an|the) \w+', r'\b\w+ like \w+',
                r'\bas \w+ as \w+', r'\b\w+ (?:burns|bleeds|sings|cries|screams|whispers|dies|lives)']:
        found.extend(re.findall(pat, text, re.IGNORECASE)[:3])
    return found[:8]


def score_metrics(text, metrics):
    tl = text.lower()
    scores = {}
    for k, cfg in metrics.items():
        raw = measure_iambic(text) if k == "iambic_rhythm" else min(
            sum(1 for sig in cfg["signals"] if sig in tl) / max(len(cfg["signals"])*0.3, 1), 1.0)
        scores[k] = {"raw": round(raw,3), "weighted": round(min(raw*cfg["weight"],1.0),3), "desc": cfg["desc"]}
    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  AI ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

PROMPT = """You are a dual creative genius inhabiting TWO minds simultaneously:

MIND 1 — CHRISTOPHER NOLAN
Think in: temporal recursion, psychological obsession, moral paradox, visual subtext, structural echoes, weight of memory.

MIND 2 — WILLIAM SHAKESPEARE
Think in: iambic heartbeats, soliloquy, tragic flaws, dramatic irony, love/duty collision, metaphors that bleed.

SUBTITLE DATA FOR THIS SCENE:
{subtitle_context}

Score each dimension 1-10. Return ONLY valid JSON (no markdown, no preamble):
{{
  "scene_title": "A poetic title for this scene",
  "scene_essence": "One sentence capturing the soul of this scene",
  "nolan_analysis": {{
    "what_nolan_sees": "What his eye catches first",
    "temporal_complexity": {{"score": 7, "note": "..."}},
    "psychological_depth": {{"score": 8, "note": "..."}},
    "moral_ambiguity": {{"score": 6, "note": "..."}},
    "structural_mirroring": {{"score": 5, "note": "..."}},
    "visual_subtext": {{"score": 7, "note": "..."}},
    "philosophical_tension": {{"score": 6, "note": "..."}},
    "nolan_improvements": [
      {{"problem": "what is hollow", "solution": "how Nolan fixes it", "example_rewrite": "concrete rewritten line/direction"}}
    ],
    "nolan_overall": "His final verdict"
  }},
  "shakespeare_analysis": {{
    "what_shakespeare_sees": "What the Bard's eye catches",
    "iambic_rhythm": {{"score": 5, "note": "..."}},
    "dramatic_irony": {{"score": 7, "note": "..."}},
    "soliloquy_potential": {{"score": 8, "note": "..."}},
    "conflict_polarity": {{"score": 6, "note": "..."}},
    "poetic_density": {{"score": 5, "note": "..."}},
    "tragic_flaw_visibility": {{"score": 7, "note": "..."}},
    "catharsis_arc": {{"score": 6, "note": "..."}},
    "shakespeare_improvements": [
      {{"problem": "what is flat", "solution": "how Shakespeare elevates it", "example_rewrite": "rewritten passage in dramatic style"}}
    ],
    "soliloquy": "6-10 line soliloquy this scene's protagonist should deliver",
    "shakespeare_overall": "His final verdict"
  }},
  "subtitle_observations": {{
    "silence_as_subtext": "How the pauses and silences carry meaning in this scene",
    "dialogue_rhythm_note": "What pace and density reveal about character state",
    "speaker_dynamic": "What the speaker balance (who talks / who goes silent) reveals"
  }},
  "metaphors_detected": ["list of metaphors/images found"],
  "missed_metaphors": ["metaphors this scene could have used but didn't"],
  "poetry_found": "Any poetic language or rhythm already present",
  "poetry_missing": "What poetic dimension is absent",
  "combined_verdict": {{
    "overall_score": 7,
    "diagnosis": "Core problem in one sentence",
    "the_rewrite": "Improved version of the key moment — Nolan depth + Shakespeare poetry",
    "what_to_measure_next": ["3-4 specific craft elements to track on revision"]
  }}
}}

SCENE TEXT:
{scene_text}
"""


def call_ollama(scene_text, subtitle_context, model="llama3.1:8b"):
    try:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": PROMPT.format(
                subtitle_context=subtitle_context, scene_text=scene_text[:5000]),
                  "stream": False, "format": "json"},
            timeout=300
        )
        if resp.status_code == 200:
            return json.loads(resp.json().get("response", "{}"))
    except Exception as e:
        log(f"[yellow]Ollama unavailable: {e}[/yellow]" if RICH else f"Ollama unavailable: {e}")
    return None


def nlp_fallback(scene_text, subtitle_context):
    ns = score_metrics(scene_text, NOLAN_METRICS)
    ss = score_metrics(scene_text, SHAKESPEARE_METRICS)
    return {
        "scene_title": "Scene Under Analysis",
        "scene_essence": "Install Ollama for full Nolan × Shakespeare analysis.",
        "nolan_analysis": {
            "what_nolan_sees": "NLP scan — install Ollama for Nolan's full vision.",
            **{k: {"score": round(v["weighted"]*10,1), "note": v["desc"]} for k,v in ns.items()},
            "nolan_improvements": [{"problem": "LLM needed for deep analysis",
                "solution": "ollama pull llama3.1:8b  then re-run", "example_rewrite": ""}],
            "nolan_overall": "Run with Ollama for Nolan's full verdict."
        },
        "shakespeare_analysis": {
            "what_shakespeare_sees": f"Iambic score: {round(measure_iambic(scene_text),2)}",
            **{k: {"score": round(v["weighted"]*10,1), "note": v["desc"]} for k,v in ss.items()},
            "shakespeare_improvements": [{"problem": "LLM needed for poetic analysis",
                "solution": "ollama pull llama3.1:8b  then re-run", "example_rewrite": ""}],
            "soliloquy": "To speak or stay silent — that is the question.\nFor in the silence lives all that we dare not name.",
            "shakespeare_overall": "Run with Ollama for the Bard's full verdict."
        },
        "subtitle_observations": {
            "silence_as_subtext": subtitle_context,
            "dialogue_rhythm_note": "Available in Ollama mode.",
            "speaker_dynamic": "Available in Ollama mode."
        },
        "metaphors_detected": detect_metaphors(scene_text),
        "missed_metaphors": ["time as water", "silence as weight", "choice as crossroads"],
        "poetry_found": f"Iambic rhythm: {round(measure_iambic(scene_text)*10,1)}/10",
        "poetry_missing": "Extended metaphor, volta, dramatic apostrophe",
        "combined_verdict": {
            "overall_score": 5,
            "diagnosis": "NLP fallback — install Ollama for full analysis",
            "the_rewrite": "Install Ollama: ollama pull llama3.1:8b, then re-run.",
            "what_to_measure_next": [
                "Temporal layering — does the past haunt the present moment?",
                "Iambic heartbeat — does dialogue carry natural rhythm?",
                "Subtext density — what is NOT being said?",
                "Cathartic potential — where is the moment of recognition?"
            ]
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def bar(score, mx=10):
    f = int((score/mx)*12)
    return f"[{'█'*f}{'░'*(12-f)}] {score:.1f}/10"


def display_subtitle_stats(stats, scene_info):
    if not RICH or not stats:
        return
    t = Table(box=box.SIMPLE, show_header=False, padding=(0,1))
    t.add_column("", style="cyan", width=28)
    t.add_column("")
    t.add_row("Pace", f"{stats.get('pace','')} ({stats.get('words_per_minute',0)} wpm)")
    t.add_row("Duration / lines", f"{round(scene_info.get('duration',0),1)}s  ·  {scene_info.get('line_count',0)} subtitle lines")
    t.add_row("Avg silence between lines", f"{stats.get('average_gap_seconds',0)}s")
    for gap, text in stats.get("significant_silences",[])[:2]:
        t.add_row(f"  Silence {gap}s before", f'"{text[:55]}"')
    speakers = stats.get("speakers", {})
    if len(speakers) > 1:
        for spk, wc in sorted(speakers.items(), key=lambda x: -x[1])[:4]:
            t.add_row(f"  Speaker: {spk}", f"{wc} words")
    top = stats.get("top_repeated_words", [])
    if top:
        t.add_row("Motif words", ", ".join(f"{w}×{c}" for w,c in top[:5]))
    t.add_row("Emotional register", f"{stats.get('question_lines',0)} questions · {stats.get('exclamation_lines',0)} exclamations")
    console.print(Panel(t, title="📊 Subtitle Analytics", border_style="cyan"))


def display_analysis(scene_num, scene_info, analysis, stats):
    if not RICH:
        print(json.dumps(analysis, indent=2))
        return

    console.print()
    console.rule(f"[bold gold1]✦  SCENE {scene_num}: {analysis.get('scene_title','Untitled')}  ✦[/bold gold1]", style="gold1")
    console.print(Panel(f"[italic white]{analysis.get('scene_essence','')}[/italic white]",
                        border_style="dim white", padding=(0,2)))

    display_subtitle_stats(stats, scene_info)

    nolan = analysis.get("nolan_analysis", {})
    shakes = analysis.get("shakespeare_analysis", {})

    # Nolan table
    nt = Table(box=box.SIMPLE, show_header=False, padding=(0,1))
    nt.add_column("Metric", style="cyan", width=22)
    nt.add_column("Score", width=22)
    nt.add_column("Note", style="dim")
    for label, key in [("Temporal Complexity","temporal_complexity"),("Psychological Depth","psychological_depth"),
                       ("Moral Ambiguity","moral_ambiguity"),("Structural Mirroring","structural_mirroring"),
                       ("Visual Subtext","visual_subtext"),("Philosophical Tension","philosophical_tension")]:
        d = nolan.get(key, {})
        sc = d.get("score", 0) if isinstance(d, dict) else 0
        note = d.get("note", "") if isinstance(d, dict) else ""
        col = "green" if sc >= 7 else "yellow" if sc >= 4 else "red"
        nt.add_row(label, f"[{col}]{bar(sc)}[/{col}]", note[:58])
    console.print(Panel(nt, title="[bold blue]🎬 Christopher Nolan Lens[/bold blue]",
                        subtitle=f"[dim]{nolan.get('what_nolan_sees','')[:80]}[/dim]", border_style="blue"))
    for imp in nolan.get("nolan_improvements", []):
        console.print(Panel(
            f"[red]⚠ PROBLEM:[/red] {imp.get('problem','')}\n\n"
            f"[green]✦ NOLAN SOLUTION:[/green] {imp.get('solution','')}\n\n"
            f"[cyan]REWRITE:[/cyan] [italic]{imp.get('example_rewrite','')}[/italic]",
            title="[blue]Nolan's Direction[/blue]", border_style="dim blue"))
    console.print(f"  [bold blue]Verdict:[/bold blue] [italic]{nolan.get('nolan_overall','')}[/italic]\n")

    # Shakespeare table
    st = Table(box=box.SIMPLE, show_header=False, padding=(0,1))
    st.add_column("Metric", style="magenta", width=22)
    st.add_column("Score", width=22)
    st.add_column("Note", style="dim")
    for label, key in [("Iambic Rhythm","iambic_rhythm"),("Dramatic Irony","dramatic_irony"),
                       ("Soliloquy Potential","soliloquy_potential"),("Conflict Polarity","conflict_polarity"),
                       ("Poetic Density","poetic_density"),("Tragic Flaw Visible","tragic_flaw_visibility"),
                       ("Catharsis Arc","catharsis_arc")]:
        d = shakes.get(key, {})
        sc = d.get("score", 0) if isinstance(d, dict) else 0
        note = d.get("note", "") if isinstance(d, dict) else ""
        col = "green" if sc >= 7 else "yellow" if sc >= 4 else "red"
        st.add_row(label, f"[{col}]{bar(sc)}[/{col}]", note[:58])
    console.print(Panel(st, title="[bold magenta]🎭 William Shakespeare Lens[/bold magenta]",
                        subtitle=f"[dim]{shakes.get('what_shakespeare_sees','')[:80]}[/dim]", border_style="magenta"))

    soliloquy = shakes.get("soliloquy", "")
    if soliloquy:
        console.print(Panel(f"[italic yellow]{soliloquy}[/italic yellow]",
                            title="[magenta]✦ The Soliloquy This Scene Needs[/magenta]", border_style="yellow"))
    for imp in shakes.get("shakespeare_improvements", []):
        console.print(Panel(
            f"[red]⚠ PROBLEM:[/red] {imp.get('problem','')}\n\n"
            f"[green]✦ SHAKESPEARE'S SOLUTION:[/green] {imp.get('solution','')}\n\n"
            f"[yellow]REWRITE:[/yellow] [italic]{imp.get('example_rewrite','')}[/italic]",
            title="[magenta]The Bard's Direction[/magenta]", border_style="dim magenta"))
    console.print(f"  [bold magenta]Verdict:[/bold magenta] [italic]{shakes.get('shakespeare_overall','')}[/italic]\n")

    # Subtitle observations
    sub_obs = analysis.get("subtitle_observations", {})
    if any(v for v in sub_obs.values()):
        console.print(Panel(
            f"[cyan]Silence as subtext:[/cyan] {sub_obs.get('silence_as_subtext','')}\n"
            f"[cyan]Dialogue rhythm:[/cyan] {sub_obs.get('dialogue_rhythm_note','')}\n"
            f"[cyan]Speaker dynamic:[/cyan] {sub_obs.get('speaker_dynamic','')}",
            title="🔇 Subtitle-Derived Observations", border_style="dim cyan"))

    # Metaphors + poetry
    console.print(Panel(
        f"[green]FOUND:[/green] {', '.join(analysis.get('metaphors_detected',[])[:5]) or 'None'}\n"
        f"[red]MISSED:[/red] {', '.join(analysis.get('missed_metaphors',[])[:5]) or 'None'}\n\n"
        f"[cyan]POETRY PRESENT:[/cyan] {analysis.get('poetry_found','')}\n"
        f"[yellow]POETRY ABSENT:[/yellow] {analysis.get('poetry_missing','')}",
        title="🔮 Metaphor & Poetry Audit", border_style="cyan"))

    # Combined verdict
    v = analysis.get("combined_verdict", {})
    sc = v.get("overall_score", 0)
    col = "green" if sc >= 7 else "yellow" if sc >= 4 else "red"
    console.print(Panel(
        f"[{col}]OVERALL: {bar(sc)}[/{col}]\n\n"
        f"[bold]DIAGNOSIS:[/bold] {v.get('diagnosis','')}\n\n"
        f"[bold gold1]✦ THE REWRITE:[/bold gold1]\n[italic white]{v.get('the_rewrite','')}[/italic white]\n\n"
        f"[bold]WHAT TO MEASURE NEXT:[/bold]\n" +
        "\n".join(f"  → {m}" for m in v.get("what_to_measure_next",[])),
        title="[bold white]✦ Combined Verdict: Nolan × Shakespeare[/bold white]", border_style="bright_white"))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🎬🎭 Scene Improvement Engine v2 — Subtitle-Aware · Nolan × Shakespeare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subtitle source priority:
  1. --subtitle flag (explicit file)
  2. External .srt / .vtt / .ass next to the video (auto-detected)
  3. Embedded subtitle track in the video container (via ffmpeg)
  4. Whisper speech-to-text fallback (always works, no subtitle file needed)

Examples:
  python scene_improver.py movie.mp4
  python scene_improver.py movie.mkv --subtitle movie.en.srt
  python scene_improver.py movie.mp4 --whisper-model small --scenes 5
  python scene_improver.py --text "She said nothing. He didn't turn around."
  python scene_improver.py movie.mp4 --dump-subtitles     # inspect cues
  python scene_improver.py movie.mp4 --scene-gap 8        # bigger silence = new scene
        """
    )
    parser.add_argument("video", nargs="?", help="Path to video file (mp4, mkv, avi, mov, etc.)")
    parser.add_argument("--subtitle", help="Explicit subtitle file (.srt / .vtt / .ass)")
    parser.add_argument("--text", help="Analyze a text passage directly (no video needed)")
    parser.add_argument("--transcript", help="Path to a plain text transcript file")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny","base","small","medium","large"],
                        help="Whisper model (default: base). Use 'small' or 'medium' for better accuracy.")
    parser.add_argument("--ollama-model", default="llama3.1:8b",
                        help="Ollama model for analysis (default: llama3.1:8b)")
    parser.add_argument("--scenes", type=int, default=3,
                        help="Max number of scenes to analyze (default: 3)")
    parser.add_argument("--scene-gap", type=float, default=5.0,
                        help="Silence gap in seconds that marks a scene boundary (default: 5.0)")
    parser.add_argument("--output", default="scene_improvements.json",
                        help="Output JSON file (default: scene_improvements.json)")
    parser.add_argument("--dump-subtitles", action="store_true",
                        help="Print all extracted subtitle cues then exit (for inspection)")
    args = parser.parse_args()

    if RICH:
        console.print(Panel(
            "[bold gold1]🎬 Scene Improvement Engine v2[/bold gold1]\n"
            "[dim]Subtitle-aware · [cyan]Christopher Nolan[/cyan] × [magenta]William Shakespeare[/magenta][/dim]\n"
            "[dim italic]Silence · Depth · Rhythm · Subtext · Catharsis · Poetry[/dim italic]",
            border_style="gold1", padding=(1,4)
        ))

    subs = []
    scenes_raw = []

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Determine input source ─────────────────────────────────────────

        if args.text:
            scenes_raw = [{"subs":[], "text": args.text, "start":0, "end":0, "duration":0, "line_count":0}]

        elif args.transcript:
            text = Path(args.transcript).read_text(encoding="utf-8")
            scenes_raw = [{"subs":[], "text": text, "start":0, "end":0, "duration":0, "line_count":0}]

        elif args.video:
            if not os.path.exists(args.video):
                log(f"[red]File not found: {args.video}[/red]" if RICH else f"Not found: {args.video}")
                sys.exit(1)

            sub_path = None

            # Priority 1: explicit --subtitle flag
            if args.subtitle:
                sub_path = args.subtitle
                log(f"\n[cyan]① Using supplied subtitle file: {sub_path}[/cyan]" if RICH
                    else f"\n① Subtitle: {sub_path}")

            # Priority 2: external file alongside video
            if not sub_path:
                log("\n[bold cyan]① Looking for external subtitle file...[/bold cyan]" if RICH
                    else "\n① Checking for external subtitle...")
                sub_path = find_external_subtitle(args.video)
                if not sub_path:
                    log("[dim]  None found.[/dim]" if RICH else "  None found.")

            # Priority 3: embedded subtitle track
            if not sub_path:
                log("[bold cyan]② Checking for embedded subtitle track...[/bold cyan]" if RICH
                    else "② Checking embedded track...")
                sub_path = extract_embedded_subtitles(args.video, tmpdir)

            # Priority 4: Whisper
            if not sub_path:
                log("[bold cyan]③ No subtitle file — using Whisper speech-to-text...[/bold cyan]" if RICH
                    else "③ Whisper fallback...")
                subs = whisper_timed_segments(args.video, args.whisper_model, tmpdir)
            else:
                subs = load_subtitle_file(sub_path)

            if not subs:
                log("[red]Could not obtain any subtitle or transcript data. Exiting.[/red]"
                    if RICH else "No data. Exiting.")
                sys.exit(1)

            if args.dump_subtitles:
                log("\n[bold]All subtitle cues:[/bold]" if RICH else "\nAll subtitle cues:")
                for s in subs:
                    print(repr(s))
                return

            log(f"\n[bold cyan]④ Segmenting scenes (silence gap ≥ {args.scene_gap}s = new scene)...[/bold cyan]"
                if RICH else f"\n④ Segmenting...")
            scenes_raw = segment_by_subtitles(subs, scene_gap=args.scene_gap)
            log(f"[green]→ {len(scenes_raw)} scene(s) detected.[/green]" if RICH
                else f"→ {len(scenes_raw)} scene(s).")
        else:
            parser.print_help()
            sys.exit(0)

        # ── Analyze scenes ─────────────────────────────────────────────────

        to_analyze = scenes_raw[:args.scenes]
        all_results = []

        for i, scene_info in enumerate(to_analyze, 1):
            start_s = fmt_time(scene_info.get("start", 0))
            end_s = fmt_time(scene_info.get("end", 0))
            log(f"\n[bold cyan]⑤ Analyzing Scene {i}/{len(to_analyze)}  [{start_s} → {end_s}]...[/bold cyan]"
                if RICH else f"\n⑤ Scene {i}/{len(to_analyze)} [{start_s}]...")

            stats = analyze_subtitle_structure(scene_info["subs"]) if scene_info["subs"] else {}
            subtitle_context = build_subtitle_context(stats, scene_info)
            scene_text = scene_info["text"]

            if not scene_text.strip():
                continue

            analysis = call_ollama(scene_text, subtitle_context, args.ollama_model)
            if not analysis:
                log("[yellow]Using NLP fallback (install Ollama for full analysis)[/yellow]"
                    if RICH else "NLP fallback.")
                analysis = nlp_fallback(scene_text, subtitle_context)

            display_analysis(i, scene_info, analysis, stats)
            all_results.append({
                "scene": i,
                "start": start_s,
                "end": end_s,
                "duration_seconds": round(scene_info.get("duration",0), 1),
                "line_count": scene_info.get("line_count", 0),
                "subtitle_stats": stats,
                "text_excerpt": scene_text[:400],
                "analysis": analysis
            })

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        log(f"\n[bold green]📄 Report saved to:[/bold green] {args.output}" if RICH
            else f"\n📄 Saved: {args.output}")


if __name__ == "__main__":
    main()