from carnatic_melody_synthesis.synthesis import Synthesizer




# Get freq limits to compute minf0
tmp_est_freq = [x for x in est_freq if x > 20]
if len(tmp_est_freq) > 0:
    minf0 = min(tmp_est_freq) - 20
else:
    minf0 = 0

# Synthesize vocal track
synthesizer = Synthesizer(
    model='hpr',
    minf0=minf0,
    maxf0=max(pitch_processed) + 50,
)
synthesized_audio, pitch_track = synthesizer.synthesize(
    filtered_audio=audio,
    pitch_track=pitch_processed,
)