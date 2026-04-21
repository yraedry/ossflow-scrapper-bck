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
    # temperature 0.65: compromiso entre voz neutra (baja temp = menos
    # acentos raros, timbre más cerca de la ref) y evitar EOS premature.
    # El synthesizer tiene retry escalado: si la primera pasada truncó,
    # sube a 0.80 y luego 0.95 para variar tokens y esquivar el bug ES
    # de Chatterbox (tokens comunes 6405/6324/4137/1034 disparan
    # "repetition detection"). Así obtenemos voz limpia en el caso
    # normal y cobertura anti-truncamiento sólo cuando hace falta.
    tts_temperature: float = 0.65
    target_language: str = "es"
    # char_limit 260: menos cortes mid-frase. Cada chunk reinicia prosodia en
    # Chatterbox, y juntar chunks con cross-fade suena robótico. Mejor una
    # sola pasada por frase siempre que quepa.
    tts_char_limit: int = 260
    tts_crossfade_ms: int = 120
    # Inter-phrase crossfade: short fade-out at the tail of TTS N and
    # fade-in at the head of TTS N+1 when the real gap between them is
    # small (< inter_phrase_crossfade_max_gap_ms). Smooths the classic
    # "phrase. [silence] phrase." boundary that Chatterbox produces when
    # every SRT block is synthesized independently.
    inter_phrase_crossfade_ms: int = 100
    inter_phrase_crossfade_max_gap_ms: int = 300
    # Stronger crossfade for boundaries the overlap-resolver had to touch
    # (two Chatterbox renders butted up against each other). 180 ms is
    # enough for a smooth click-free join without muting hard consonants
    # at phrase starts.
    force_crossfade_ms: int = 180
    # exaggeration 0.30: coach plano, menos teatral. Baja variación de
    # énfasis entre chunks → menos "acentos raros" percibidos.
    tts_exaggeration: float = 0.30
    # repetition_penalty 1.45: compensa la bajada de temperature a 0.65.
    # Con menos variedad del sampler necesitamos penalizar más fuerte la
    # repetición real para esquivar el bug ES de Chatterbox (tokens
    # comunes 6405/6324/4137/1034 disparando EOS premature). Si la
    # primera pasada aún trunca, el retry sube penalty a 1.55/1.65.
    tts_repetition_penalty: float = 1.45
    # min_p/top_p abiertos: 0.02/1.0 permite al sampler escoger tokens menos
    # probables cuando los top-k repiten (rompe ciclos de repetición que
    # disparan EOS prematuro en Chatterbox ES).
    tts_min_p: float = 0.02
    tts_top_p: float = 1.0
    # Legacy field (unused with Chatterbox but kept for back-compat)
    tts_model_name: str = "chatterbox-multilingual"

    tts_engine: str = "xttsv2"

    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_config_path: str = ""
    xtts_checkpoint_dir: str = ""
    xtts_use_deepspeed: bool = False
    xtts_code_switching: bool = True
    xtts_en_terms_extra: tuple[str, ...] = ()

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
    # Mantener ducking sostenido si hay una pausa corta entre frases. Evita
    # el efecto "frenazo" cuando el TTS sale más corto que el slot SRT: sin
    # esto, el background sube a full volume durante el hueco y baja de
    # golpe al arrancar la siguiente frase. Mergeamos regiones de ducking
    # separadas por <=1500 ms para que el background quede bajo durante
    # pausas breves, simulando el comportamiento de un coach real que
    # pausa unos segundos entre explicaciones.
    ducking_sustain_gap_ms: int = 1500

    # Sync / alignment
    lookahead_phrases: int = 5
    min_phrase_duration_ms: int = 600
    avg_ms_per_char: float = 65.0       # español ~65 ms/char
    inter_phrase_pad_ms: int = 20       # pausa mínima entre frases — prosodia más continua
    tts_lead_silence_ms: int = 0        # sin silencio inicial — encadena con cola de la frase anterior

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

    # Compactado "sintético" con guía de words.json: cuando el TTS generado
    # acaba bastante antes del siguiente anchor, normalmente es porque
    # Chatterbox ha clonado un ritmo más rápido que el speaker EN original
    # → el slot tiene silencio que NO existe en el audio original. Si
    # words.json confirma que el speaker seguía hablando en ese tramo,
    # cerramos el hueco desplazando la siguiente frase. Si no hay palabras
    # en el gap (pausa real del coach: cámara, gesto, respiración larga),
    # dejamos el silencio intacto para preservar lip sync.
    compact_synthetic_silence: bool = True
    # Umbral a partir del cual consideramos que un hueco es "anómalo" y
    # merece revisión contra words.json. Por debajo dejamos al mixer
    # resolverlo con ducking sostenido. 600 ms evita tocar pausas
    # naturales cortas (inhalación, coma prosódica).
    compact_synthetic_silence_threshold_ms: int = 600
    # Fracción mínima del gap anómalo que words.json debe cubrir con
    # habla para que consideremos el gap sintético. 0.70 = al menos 70%
    # del hueco tiene palabras del speaker → cerrar. Por debajo, asumimos
    # que es una pausa real aunque haya alguna palabra suelta al borde.
    compact_synthetic_min_speech_ratio: float = 0.70
    # Pad mínimo entre frases tras el compactado (aire natural). 200 ms
    # también enmascara mejor los saltos RMS entre frases Chatterbox que
    # quedarían pegadas con un pad demasiado corto.
    compact_synthetic_min_gap_ms: int = 200
    # Safety rail de lip sync: ninguna frase puede acabar anclada más de
    # este margen antes de su anchor original (el ``original_start_ms``
    # que heredó de la alineación SRT/words). Si cerrar un gap requiere
    # mover la siguiente frase más de esto, limitamos el shift y el
    # silencio restante se deja tal cual — preferimos un silencio
    # intermedio a un desfase de lip sync perceptible. Cascada: una vez
    # una frase queda capada, las siguientes arrastran el mismo offset
    # efectivo respecto a su anchor.
    compact_synthetic_max_drift_ms: int = 400
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
