"""Procedural lo-fi soundtrack + foley for a tutorial Short (numpy only).
Usage: python3 audio/soundtrack.py events.json out.wav"""
import json, sys, wave
import numpy as np

SR = 44100
ev = json.load(open(sys.argv[1]))
T = ev['duration'] + 0.5
N = int(T * SR)
rng = np.random.default_rng(3)
t = np.arange(N) / SR
mix = np.zeros(N)

def place(buf, at, sig, gain=1.0):
    i = int(at * SR)
    if i >= len(buf): return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain

def lowpass(x, a):  # one-pole
    y = np.empty_like(x); acc = 0.0
    for k in range(len(x)):
        acc += a * (x[k] - acc); y[k] = acc
    return y

def ep_note(f, dur):  # soft electric-piano tone
    n = int(dur * SR); tt = np.arange(n) / SR
    env = np.exp(-tt * 2.2) * np.minimum(1, tt / 0.012)
    trem = 1 + 0.08 * np.sin(2 * np.pi * 4.5 * tt)
    s = np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * 2 * f * tt + 0.3) * np.exp(-tt * 4) + 0.12 * np.sin(2 * np.pi * 3.01 * f * tt) * np.exp(-tt * 7)
    return s * env * trem

midi = lambda m: 440 * 2 ** ((m - 69) / 12)
bpm = 84; beat = 60 / bpm; bar = 4 * beat
chords = [[53, 57, 60, 64], [52, 55, 59, 62], [50, 53, 57, 60], [48, 52, 55, 59]]  # Fmaj7 Em7 Dm7 Cmaj7
music = np.zeros(N)
nb = int(T / bar) + 1
for b in range(nb):
    ch = chords[b % 4]
    t0 = b * bar
    for k, m in enumerate(ch):  # gentle strum
        place(music, t0 + k * 0.025, ep_note(midi(m), bar * 0.98), 0.16)
    place(music, t0 + 2 * beat + 0.01, ep_note(midi(ch[0] - 12), bar * 0.5), 0.18)
    # simple melody: chord tones an octave up on off-beats
    for s, d in [(1.5, 0.5), (2.5, 0.5), (3.0, 1.0)]:
        if rng.random() < 0.7:
            place(music, t0 + s * beat, ep_note(midi(rng.choice(ch) + 12), d * beat * 1.6), 0.07)
    for q in range(4):  # soft kick on 1 and 3, brushed hat on 8ths
        if q % 2 == 0:
            n = int(0.25 * SR); tt = np.arange(n) / SR
            kick = np.sin(2 * np.pi * (55 + 70 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 14)
            place(music, t0 + q * beat, kick, 0.32)
        for h in (0, 0.5):
            n = int(0.06 * SR); nz = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 70)
            nz = nz - lowpass(nz, 0.25)  # high-pass-ish
            place(music, t0 + (q + h) * beat, nz, 0.025 if h else 0.035)
music = lowpass(music, 0.35)
# fade in/out
fade = np.minimum(1, t / 1.5) * np.minimum(1, (T - t) / 2.0)
music *= fade
music *= 10 ** (-20 / 20) / (np.sqrt(np.mean(music ** 2)) + 1e-9)  # ~ -20 dBFS RMS

sfx = np.zeros(N)
def click(fast):
    n = int(0.05 * SR); tt = np.arange(n) / SR
    s = np.sin(2 * np.pi * 2400 * tt) * np.exp(-tt * 140) + 0.5 * rng.standard_normal(n) * np.exp(-tt * 300)
    return s * (0.5 if fast else 0.9)
for c in ev['clicks']:
    place(sfx, c['t'], click(c['fast']), 0.12)
def whoosh():
    n = int(0.45 * SR); tt = np.arange(n) / SR
    nz = lowpass(rng.standard_normal(n), 0.08)
    return nz * np.sin(np.pi * tt / tt[-1]) ** 2
for s in ev['steps'][1:]:
    place(sfx, s - 0.15, whoosh(), 0.5)
def pop():
    n = int(0.18 * SR); tt = np.arange(n) / SR
    return np.sin(2 * np.pi * (500 + 900 * tt / tt[-1]) * tt) * np.exp(-tt * 18)
place(sfx, 0.05, pop(), 0.25)
place(sfx, ev['steps'][-1] + 1.3, pop(), 0.25)

out = music + sfx
out /= max(1.0, np.max(np.abs(out)) / 0.95)
pcm = (out * 32767).astype(np.int16)
st = np.stack([pcm, pcm], axis=1)
with wave.open(sys.argv[2], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(st.tobytes())
print('wrote', sys.argv[2], round(T, 1), 's')
