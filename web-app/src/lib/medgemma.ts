export interface ContributionItem {
  feature: string
  delta: number
  direction: 'risk' | 'protective'
}

export interface InterpretRequest {
  patient_name: string
  age: number
  sex: string
  bacteriologic_status: string
  microscopy_result: string
  anatomical_site: string
  registration_group: string
  source_of_patient: string
  type: string
  days_to_treatment: number
  failure_probability: number
  contributions: ContributionItem[]
}

export function streamInterpretation(
  request: InterpretRequest,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (error: Error) => void,
): AbortController {
  const controller = new AbortController()

  ;(async () => {
    let response: Response
    try {
      response = await fetch('/api/interpret', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      })
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)))
      return
    }

    if (!response.ok) {
      onError(new Error(response.statusText))
      return
    }

    if (!response.body) {
      onError(new Error('Response body is null'))
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        // Keep the last (potentially incomplete) line in the buffer
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice('data: '.length).trim()
          if (!raw) continue

          let parsed: { token?: string; done?: boolean; error?: string }
          try {
            parsed = JSON.parse(raw)
          } catch {
            continue
          }

          if (parsed.error) {
            onError(new Error(parsed.error))
            return
          }

          if (parsed.done === true) {
            onDone()
            return
          }

          if (parsed.token && parsed.token.length > 0) {
            onToken(parsed.token)
          }
        }
      }
    } catch (err) {
      // Ignore abort errors
      if (err instanceof Error && err.name === 'AbortError') return
      onError(err instanceof Error ? err : new Error(String(err)))
    }
  })()

  return controller
}
