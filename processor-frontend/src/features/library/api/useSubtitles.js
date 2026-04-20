// Subtitle validation + per-segment regeneration hooks.
// Proxies to processor-api /api/subtitles/*.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { http } from '@/lib/httpClient'
import { qk } from '@/lib/queryKeys'

export function useValidateSubs(srtPath, { enabled = true } = {}) {
  return useQuery({
    queryKey: qk.subtitles.validate(srtPath),
    queryFn: () => http.post('/subtitles/validate', { srt_path: srtPath }),
    enabled: Boolean(srtPath) && enabled,
    staleTime: 30_000,
  })
}

export function useRegenerateSegment() {
  return useMutation({
    mutationFn: ({ srtPath, segmentIdx, contextSeconds = 1.0, videoPath }) =>
      http.post('/subtitles/regenerate-segment', {
        srt_path: srtPath,
        segment_idx: segmentIdx,
        context_seconds: contextSeconds,
        video_path: videoPath,
      }),
  })
}

export function useApplySegment() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ srtPath, segmentIdx, text, start, end }) =>
      http.post('/subtitles/apply-segment', {
        srt_path: srtPath,
        segment_idx: segmentIdx,
        text,
        start,
        end,
      }),
    onSuccess: (_d, vars) => {
      qc.invalidateQueries({ queryKey: qk.subtitles.validate(vars.srtPath) })
    },
  })
}

export function useTranslateSrt() {
  return useMutation({
    mutationFn: ({ srtPath, sourceLang = 'EN', targetLang = 'ES', formality, dubbingMode = false }) =>
      http.post('/subtitles/translate', {
        srt_path: srtPath,
        source_lang: sourceLang,
        target_lang: targetLang,
        formality,
        dubbing_mode: dubbingMode,
      }),
  })
}

export function useAnalyzeVideo() {
  return useMutation({
    mutationFn: ({ videoPath, language = 'en', model = 'large-v3' }) =>
      http.post('/subtitles/analyze', {
        video_path: videoPath,
        language,
        model,
      }),
  })
}
