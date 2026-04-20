"""Centralized configuration for the dubbing generator."""

from dataclasses import dataclass


@dataclass
class DubbingConfig:
    """All dubbing pipeline parameters in one place."""

    # TTS (Chatterbox Multilingual)
    # Valores orientados a ES natural sobre ref EN: baja guidance (cfg_weight
    # 0.25-0.3 en synthesizer) desata Chatterbox del ritmo EN de la ref y
    # deja salir prosodia ES propia. Mantenemos timbre porque eso depende
    # del audio prompt, no del cfg_weight.
    # 0.88: sweet spot validado auditivamente para coach pausado (Jozef
    # Chen). 0.85 deja huecos. 0.92 suena raro/rápido. 0.88 puede
    # extender el dub pasado el final del vídeo — se compensa activando
    # allow_video_tail_extension (pad del último frame) en vez de acelerar.
    tts_speed: float = 0.88
    # temperature 0.80: más alta de lo habitual pero necesaria. Chatterbox
    # tiene bug documentado con ES donde tokens comunes (6405, 6324, 4137,
    # 1034 = puntuación/espacio) se repiten y disparan EOS prematuro. Subir
    # temp aumenta variedad del sampler → distintos tokens → menos falsos
    # positivos de repetition detection. Prosodia sigue coherente porque
    # cfg_weight (0.45) ya ancla el ritmo al audio de referencia.
    tts_temperature: float = 0.80
    target_language: str = "es"
    # char_limit 260: menos cortes mid-frase. Cada chunk reinicia prosodia en
    # Chatterbox, y juntar chunks con cross-fade suena robótico. Mejor una
    # sola pasada por frase siempre que quepa.
    tts_char_limit: int = 260
    tts_crossfade_ms: int = 120
    # exaggeration 0.30: coach plano, menos teatral. Baja variación de
    # énfasis entre chunks → menos "acentos raros" percibidos.
    tts_exaggeration: float = 0.30
    # repetition_penalty 1.35: 1.25 seguía disparando "Detected 2x
    # repetition" en 15/15 frases con tokens comunes ES (6405, 6324, 4137)
    # → EOS prematuro → TTS truncado hasta -2400ms → gaps audibles. 1.35
    # penaliza más fuerte la repetición real y deja pasar los falsos
    # positivos. Combinado con temperature 0.70 (más variedad).
    tts_repetition_penalty: float = 1.35
    # min_p/top_p abiertos: 0.02/1.0 permite al sampler escoger tokens menos
    # probables cuando los top-k repiten (rompe ciclos de repetición que
    # disparan EOS prematuro en Chatterbox ES).
    tts_min_p: float = 0.02
    tts_top_p: float = 1.0
    # Legacy field (unused with Chatterbox but kept for back-compat)
    tts_model_name: str = "chatterbox-multilingual"

    # Voice cloning
    # Chatterbox cloning quality peaks at 8-15 s of clean ref speech. Going
    # longer (the old 30 s) dilutes timbre coherence and increases the chance
    # of capturing ref disfluencies that leak into the ES output.
    voice_sample_duration: float = 12.0
    use_model_voice: bool = False
    model_voice_path: str = ""

    # Audio stretching — SRT ES literal puede ser 30-40% más largo que EN.
    # Rango amplio [0.9, 1.30]: comprimimos hasta 1.30x si hace falta, pero
    # tratamos de que el traductor ya compacte lo posible y sólo aceleremos
    # puntualmente. 0.9 mínimo para rellenar huecos sin distorsionar.
    max_compression_ratio: float = 1.40
    min_compression_ratio: float = 0.85
    silence_threshold_db: float = -40.0
    silence_chunk_ms: int = 10

    # Drift correction — stretch dinámico activado con rango conservador.
    drift_check_interval: int = 8
    drift_threshold_ms: float = 250.0
    speed_base: float = 1.0
    speed_min: float = 0.95
    speed_max: float = 1.15
    constant_speed: bool = False

    # Audio ducking — voz domina pero fondo audible (ambiente natural)
    ducking_bg_volume: float = 0.12    # ~-18 dB
    ducking_fg_volume: float = 1.6     # +4 dB TTS
    ducking_fade_ms: int = 180

    # Sync / alignment
    lookahead_phrases: int = 5
    min_phrase_duration_ms: int = 600
    avg_ms_per_char: float = 65.0       # español ~65 ms/char
    inter_phrase_pad_ms: int = 80       # pausa natural entre frases
    tts_lead_silence_ms: int = 40       # micro-silencio inicial (natural attack)

    # Sync — política industria: cuadrar con timestamps originales.
    # El texto ES ya debe venir compactado por el traductor (iso-sincronía).
    # Si sobra <=200 ms, pequeño stretch imperceptible lo absorbe.
    max_overflow_ms: int = 200
    shift_subsequent_on_overflow: bool = False

    # Compact-silence: cuando el TTS sale más corto que el slot planificado,
    # si el hueco final supera este umbral, desplazamos las frases
    # posteriores hacia atrás para cerrar el silencio. Solo se activa con
    # slots anchored a habla real (nivel 3 dub track); con SRT lectura
    # mantendríamos las pausas intencionadas del subtítulo.
    # Con merge_consecutive_blocks híbrido, los gaps restantes son pausas
    # reales del speaker — no deben cerrarse o desalinean con el vídeo.
    compact_trailing_silence: bool = False
    compact_trailing_silence_threshold_ms: int = 250
    # Reservamos pad entre frases para que el compactado no pegue dos
    # frases sin aire natural. Complementa inter_phrase_pad_ms.
    compact_min_gap_ms: int = 80
    # Desactivado: el vídeo termina donde termina; no extendemos.
    allow_video_tail_extension: bool = False
    video_tail_extension_max_ms: int = 0

    # Hybrid merge (opción 3): agrupa líneas SRT consecutivas con hueco
    # pequeño en un único bloque TTS. El TTS habla con prosodia continua
    # (menos cortes robóticos) pero el bloque sigue anclado al timestamp
    # del vídeo (start = primera línea, end = última línea). Basado en
    # `.es.srt` literal alineado con vídeo, no en segmenter nivel 3.
    # Merge desactivado: Chatterbox fuerza EOS prematuro con textos largos
    # (~200 chars) al detectar repetición de tokens → TTS se corta en mitad
    # de la frase → huecos. Mantenemos 1 línea SRT = 1 TTS, más cortos y
    # robustos. Aligner + stretcher absorben sync.
    merge_consecutive_blocks: bool = False
    merge_max_gap_ms: int = 400
    merge_max_chars: int = 200

    # Paths / file discovery
    extensions: tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov")
