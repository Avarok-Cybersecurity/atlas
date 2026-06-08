# Atlas FP8 + opencode iteration log

Started: 2026-05-25T11:21:51-04:00

Latest image: atlas-gb10:fp8-fullstream
Container: atlas-qwen-final

---

## [11:22:30] Iteration 1

**Container**: atlas-qwen-final — Up 19 minutes, running
**Recent container activity** (last 5m):
- 15:06:45 Request: messages=18, tools=9, stream=true, temp=0.3 → Done: 284 tok (stop) 38.6 tok/s, TTFT=1343.8ms
- 15:06:54 Request: messages=20, tools=9, stream=true → Done: 287 tok (stop) 38.3 tok/s, TTFT=1368.5ms
- 15:07:03 Request: messages=22, tools=9, stream=true → Done: 189 tok (stop) 39.5 tok/s, TTFT=1782.0ms
- 15:08:50 Request: messages=2, tools=1, stream=false, temp=0.0 → Done: 98 tok (stop) 62.9 tok/s, TTFT=719.9ms
- 15:08:53 Request: messages=2, tools=1, stream=true, temp=0.0 → Done: 98 tok (stop) 63.9 tok/s, TTFT=719.5ms
- Last activity: 15:08:55 UTC (13+ minutes ago from now)

**Active opencode runs**: 1 process (PID 2672149), started at 11:10 local — running ~12 minutes
**Active builds**: 0
**API health**: OK — `{"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8",...}]}`

**Status**: **STALL DETECTED** — opencode running 12+ min, last container activity 13+ min ago (15:08:55 UTC)
**Stalls detected**: opencode PID 2672149 has been running since 11:10 (~12 min). No container requests since 15:08:55 UTC. Either opencode is waiting on something outside the model (filesystem, tool, network) or the session is hung.
**Notable**: Container is healthy (API responds). The opencode process is `opencode run --dangerously-skip-permissions --format json "Create /tmp/axv34/Cargo.toml and /tmp/axv34/src/main.rs for a minimal axum 0.7 ping/pong server..."` — suspiciously long runtime for a simple file-write task.


---

## [11:26:40] Iteration 1

**Container**: atlas-qwen-final Up 23 minutes
**Recent container activity** (last 2m):
- 

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779722800,"owned_by":"atlas-spark"

**Status**: No container activity in last 2m / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [11:28:10] Iteration 2

**Container**: NO ATLAS CONTAINERS
**Recent container activity** (last 2m):
- Error response from daemon: No such container: atlas-qwen-final

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 3
**API health**: 

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [11:29:40] Iteration 3

**Container**: atlas-qwen-final Up 31 seconds
**Recent container activity** (last 2m):
- 

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: 

**Status**: No container activity in last 2m / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [11:31:10] Iteration 4

**Container**: atlas-qwen-final Up 2 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:30:56.373416Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=3, tools=0, tools_active=false, tool_choice=None, stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:30:56.402644Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 2 processes
  PID 2701336 11:30 ?
  PID 2701338 11:30 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779723071,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 2
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:32:41] Iteration 5

**Container**: atlas-qwen-final Up 3 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:31:44.728672Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=6, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:32:01.704756Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 114 tokens (stop) 22.7 tok/s, TTFT=11793.3ms
- [2m2026-05-25T15:32:01.774933Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=8, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:32:16.378697Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 58 tokens (stop) 22.3 tok/s, TTFT=11977.0ms
- [2m2026-05-25T15:32:16.447606Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=10, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:32:30.589681Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 786 tokens (stop) 8.4 tok/s, TTFT=1014.3ms
- [2m2026-05-25T15:32:32.379137Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 120 tokens (stop) 32.0 tok/s, TTFT=12150.0ms
- [2m2026-05-25T15:32:32.403988Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=12, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 2 processes
  PID 2701336 11:30 ?
  PID 2701338 11:30 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779723161,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 2
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:34:11] Iteration 6

**Container**: atlas-qwen-final Up 5 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:33:37.211868Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 440 tokens (stop) 41.9 tok/s, TTFT=3454.8ms
- [2m2026-05-25T15:33:37.254410Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=22, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:33:48.307105Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 303 tokens (length) 40.9 tok/s, TTFT=3633.8ms
- [2m2026-05-25T15:33:48.340738Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=24, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:33:57.093153Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 201 tokens (stop) 39.9 tok/s, TTFT=3693.4ms
- [2m2026-05-25T15:33:57.134074Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=26, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:34:03.967536Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 114 tokens (stop) 39.7 tok/s, TTFT=3942.9ms
- [2m2026-05-25T15:34:04.005986Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=28, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 2 processes
  PID 2701336 11:30 ?
  PID 2701338 11:30 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779723251,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 2
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:35:41] Iteration 7

**Container**: atlas-qwen-final Up 6 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:35:09.852044Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=40, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:35:15.631776Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 183 tokens (stop) 38.6 tok/s, TTFT=1016.8ms
- [2m2026-05-25T15:35:15.676140Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=42, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:35:21.473836Z[0m [33m WARN[0m [2mspark::api::chat_stream::tool_handlers[0m[2m:[0m tool call validation error (stream Δ): Error: write requires a non-empty 'filePath'. Got empty string — provide an absolute path like '/tmp/calc-test75/Cargo.toml'.; replacing with content and ending [3mtool[0m[2m=[0mwrite
- [2m2026-05-25T15:35:21.499673Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 170 tokens (stop) 38.1 tok/s, TTFT=1343.2ms
- [2m2026-05-25T15:35:21.554128Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=44, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:35:27.218505Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 159 tokens (stop) 38.0 tok/s, TTFT=1453.4ms
- [2m2026-05-25T15:35:27.255153Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=46, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 2 processes
  PID 2701336 11:30 ?
  PID 2701338 11:30 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779723341,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 2
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:37:11] Iteration 8

**Container**: atlas-qwen-final Up 8 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:36:08.954440Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 252 tokens (stop) 37.7 tok/s, TTFT=1780.9ms
- [2m2026-05-25T15:36:08.989905Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=52, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:36:33.027799Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 889 tokens (stop) 40.1 tok/s, TTFT=1843.9ms
- [2m2026-05-25T15:36:33.087256Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=54, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:36:40.111023Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 188 tokens (stop) 37.8 tok/s, TTFT=2023.3ms
- [2m2026-05-25T15:36:40.147943Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=56, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:37:03.849387Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 242 tokens (stop) 37.6 tok/s, TTFT=17238.2ms
- [2m2026-05-25T15:37:03.891181Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=58, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 2 processes
  PID 2701336 11:30 ?
  PID 2701338 11:30 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779723431,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 2
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:38:41] Iteration 9

**Container**: atlas-qwen-final Up 9 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:37:54.819898Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 151 tokens (stop) 36.7 tok/s, TTFT=18004.8ms
- [2m2026-05-25T15:37:54.863445Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=66, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:38:01.599129Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 106 tokens (stop) 36.4 tok/s, TTFT=3791.1ms
- [2m2026-05-25T15:38:01.620118Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=68, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:38:11.838386Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 228 tokens (stop) 36.2 tok/s, TTFT=3893.5ms
- [2m2026-05-25T15:38:11.873319Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=70, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:38:17.907834Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 71 tokens (stop) 36.9 tok/s, TTFT=4080.1ms
- [2m2026-05-25T15:38:17.951806Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=72, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 2 processes
  PID 2701336 11:30 ?
  PID 2701338 11:30 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779723521,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 2
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:40:11] Iteration 10

**Container**: atlas-qwen-final Up 11 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:38:42.186660Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 192 tokens (stop) 36.0 tok/s, TTFT=18873.8ms
- [2m2026-05-25T15:38:42.216054Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=74, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:38:50.522404Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 138 tokens (stop) 35.8 tok/s, TTFT=4431.0ms
- [2m2026-05-25T15:38:50.541486Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=76, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:39:51.294955Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 2090 tokens (length) 37.2 tok/s, TTFT=4542.9ms
- [2m2026-05-25T15:39:51.323193Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=78, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:40:02.966314Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 250 tokens (stop) 35.7 tok/s, TTFT=4626.2ms
- [2m2026-05-25T15:40:02.990416Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=80, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 2 processes
  PID 2701336 11:30 ?
  PID 2701338 11:30 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779723612,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 2
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:41:42] Iteration 11

**Container**: atlas-qwen-final Up 12 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:40:34.445641Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 438 tokens (stop) 36.7 tok/s, TTFT=19494.0ms
- [2m2026-05-25T15:40:34.465439Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=82, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:40:41.697189Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 58 tokens (stop) 36.5 tok/s, TTFT=5620.9ms
- [2m2026-05-25T15:40:41.718181Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=84, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:41:07.857226Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 903 tokens (stop) 35.4 tok/s, TTFT=621.5ms
- [2m2026-05-25T15:41:07.904188Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=86, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:41:34.947745Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 930 tokens (stop) 35.4 tok/s, TTFT=731.7ms
- [2m2026-05-25T15:41:34.976935Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=88, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 2 processes
  PID 2701336 11:30 ?
  PID 2701338 11:30 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779723702,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 2
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:43:12] Iteration 12

**Container**: atlas-qwen-final Up 14 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:41:34.947745Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 930 tokens (stop) 35.4 tok/s, TTFT=731.7ms
- [2m2026-05-25T15:41:34.976935Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=88, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:42:04.349532Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 287 tokens (stop) 32.6 tok/s, TTFT=20546.2ms
- [2m2026-05-25T15:42:04.398027Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=90, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:42:09.798008Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 154 tokens (stop) 35.2 tok/s, TTFT=993.0ms
- [2m2026-05-25T15:42:09.839285Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=92, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:42:39.959728Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 837 tokens (length) 35.1 tok/s, TTFT=6211.0ms

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779723792,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [11:44:42] Iteration 13

**Container**: atlas-qwen-final Up 15 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:44:21.087129Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=0, tools_active=false, tool_choice=None, stream=false, temp=Some(0.0), max_tokens=400, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:44:22.586845Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 66 tokens (stop) 64.7 tok/s, TTFT=479.5ms
- [2m2026-05-25T15:44:22.588842Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=0, tools_active=false, tool_choice=None, stream=true, temp=Some(0.0), max_tokens=400, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:44:24.084720Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 66 tokens (stop) 64.8 tok/s, TTFT=476.9ms
- [2m2026-05-25T15:44:36.891001Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=0, tools_active=false, tool_choice=None, stream=false, temp=Some(0.0), max_tokens=400, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:44:37.929093Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 66 tokens (stop) 65.2 tok/s, TTFT=24.1ms
- [2m2026-05-25T15:44:37.931174Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=0, tools_active=false, tool_choice=None, stream=true, temp=Some(0.0), max_tokens=400, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:44:38.459302Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 33 tokens (stop) 65.6 tok/s, TTFT=24.5ms

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779723882,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [11:46:12] Iteration 14

**Container**: atlas-qwen-final Up 17 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:44:36.891001Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=0, tools_active=false, tool_choice=None, stream=false, temp=Some(0.0), max_tokens=400, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:44:37.929093Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 66 tokens (stop) 65.2 tok/s, TTFT=24.1ms
- [2m2026-05-25T15:44:37.931174Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=0, tools_active=false, tool_choice=None, stream=true, temp=Some(0.0), max_tokens=400, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:44:38.459302Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 33 tokens (stop) 65.6 tok/s, TTFT=24.5ms
- [2m2026-05-25T15:45:59.806602Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=0, tools_active=false, tool_choice=None, stream=false, temp=Some(0.0), max_tokens=600, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:46:01.216538Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 66 tokens (stop) 65.2 tok/s, TTFT=397.1ms
- [2m2026-05-25T15:46:01.218748Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=0, tools_active=false, tool_choice=None, stream=true, temp=Some(0.0), max_tokens=600, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:46:02.626339Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 66 tokens (stop) 65.2 tok/s, TTFT=394.9ms

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779723972,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [11:47:42] Iteration 15

**Container**: atlas-qwen-final Up 18 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:45:59.806602Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=0, tools_active=false, tool_choice=None, stream=false, temp=Some(0.0), max_tokens=600, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:46:01.216538Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 66 tokens (stop) 65.2 tok/s, TTFT=397.1ms
- [2m2026-05-25T15:46:01.218748Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=0, tools_active=false, tool_choice=None, stream=true, temp=Some(0.0), max_tokens=600, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:46:02.626339Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 66 tokens (stop) 65.2 tok/s, TTFT=394.9ms
- [2m2026-05-25T15:46:22.685818Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=3, tools=0, tools_active=false, tool_choice=None, stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:46:22.720665Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:47:07.759581Z[0m [33m WARN[0m [2mspark::api::chat_stream::tool_handlers[0m[2m:[0m tool call validation error (stream Δ): Error: Unknown tool 'bash command'. Available tools: bash, edit, glob, grep, read, skill, task, webfetch, write; replacing with content and ending [3mtool[0m[2m=[0mbash command

**Active opencode runs**: 3 processes
  PID 2710093 11:46 ?
  PID 2710675 11:46 ?
  PID 2710677 11:46 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779724062,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 3
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:49:12] Iteration 16

**Container**: atlas-qwen-final Up 20 minutes
**Recent container activity** (last 2m):
- 

**Active opencode runs**: 3 processes
  PID 2710093 11:46 ?
  PID 2710675 11:46 ?
  PID 2710677 11:46 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779724153,"owned_by":"atlas-spark"

**Status**: No container activity in last 2m / opencode procs: 3
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:50:43] Iteration 17

**Container**: atlas-qwen-final Up 21 minutes
**Recent container activity** (last 2m):
- 

**Active opencode runs**: 3 processes
  PID 2710093 11:46 ?
  PID 2710675 11:46 ?
  PID 2710677 11:46 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779724243,"owned_by":"atlas-spark"

**Status**: No container activity in last 2m / opencode procs: 3
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:52:13] Iteration 18

**Container**: atlas-qwen-final Up 23 minutes
**Recent container activity** (last 2m):
- Error response from daemon: can not get logs from container which is dead or marked for removal

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: 

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [11:53:43] Iteration 19

**Container**: atlas-qwen-final Up About a minute
**Recent container activity** (last 2m):
- 

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779724423,"owned_by":"atlas-spark"

**Status**: No container activity in last 2m / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [11:55:13] Iteration 20

**Container**: atlas-qwen-final Up 2 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:54:05.057577Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 794 tokens (stop) 55.4 tok/s, TTFT=996.3ms
- [2m2026-05-25T15:54:36.948859Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 849 tokens (stop) 41.7 tok/s, TTFT=11306.5ms
- [2m2026-05-25T15:54:36.994484Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=4, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:54:41.578644Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 111 tokens (stop) 41.8 tok/s, TTFT=1904.3ms
- [2m2026-05-25T15:54:41.611256Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=6, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:54:46.409034Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 109 tokens (stop) 39.9 tok/s, TTFT=2046.4ms
- [2m2026-05-25T15:54:46.439560Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=8, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:54:52.073862Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 145 tokens (stop) 42.3 tok/s, TTFT=2184.9ms

**Active opencode runs**: 1 processes
  PID 2720200 11:53 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779724513,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 1
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [11:56:43] Iteration 21

**Container**: atlas-qwen-final Up 4 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:54:46.409034Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 109 tokens (stop) 39.9 tok/s, TTFT=2046.4ms
- [2m2026-05-25T15:54:46.439560Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=8, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:54:52.073862Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 145 tokens (stop) 42.3 tok/s, TTFT=2184.9ms

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 4
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779724603,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [11:58:14] Iteration 22

**Container**: atlas-qwen-final Up 56 seconds
**Recent container activity** (last 2m):
- 

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: 

**Status**: No container activity in last 2m / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [11:59:44] Iteration 23

**Container**: atlas-qwen-final Up 2 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T15:58:50.274684Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=3, tools=0, tools_active=false, tool_choice=None, stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:58:50.306243Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:59:09.195767Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 127 tokens (stop) 20.8 tok/s, TTFT=11395.5ms
- [2m2026-05-25T15:59:09.298854Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=4, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:59:22.059519Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 27 tokens (stop) 24.1 tok/s, TTFT=11612.0ms
- [2m2026-05-25T15:59:22.133111Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=6, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T15:59:38.965758Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 88 tokens (stop) 23.3 tok/s, TTFT=12897.4ms
- [2m2026-05-25T15:59:39.085169Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=8, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 3 processes
  PID 2731168 11:58 ?
  PID 2731750 11:58 ?
  PID 2731752 11:58 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779724784,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 3
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [12:01:14] Iteration 24

**Container**: atlas-qwen-final Up 3 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T16:00:14.747475Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=12, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:00:33.813298Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 127 tokens (stop) 22.7 tok/s, TTFT=13439.7ms
- [2m2026-05-25T16:00:33.883984Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=14, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:00:49.208491Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 38 tokens (stop) 23.2 tok/s, TTFT=13641.5ms
- [2m2026-05-25T16:00:49.247694Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=16, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:01:08.565527Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 788 tokens (stop) 5.7 tok/s, TTFT=991.3ms
- [2m2026-05-25T16:01:09.393633Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 156 tokens (stop) 24.8 tok/s, TTFT=13810.4ms
- [2m2026-05-25T16:01:09.427886Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=18, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 3 processes
  PID 2731168 11:58 ?
  PID 2731750 11:58 ?
  PID 2731752 11:58 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779724874,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 3
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [12:02:44] Iteration 25

**Container**: atlas-qwen-final Up 5 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T16:01:45.763799Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 86 tokens (stop) 39.1 tok/s, TTFT=1083.4ms
- [2m2026-05-25T16:01:45.906870Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=26, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:01:51.803232Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 183 tokens (stop) 37.9 tok/s, TTFT=1038.1ms
- [2m2026-05-25T16:01:51.835195Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=28, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:02:14.175104Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 824 tokens (stop) 38.5 tok/s, TTFT=926.5ms
- [2m2026-05-25T16:02:14.206013Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=30, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:02:32.770096Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 662 tokens (stop) 39.3 tok/s, TTFT=1694.4ms
- [2m2026-05-25T16:02:32.819203Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=32, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 3 processes
  PID 2731168 11:58 ?
  PID 2731750 11:58 ?
  PID 2731752 11:58 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779724964,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 3
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [12:04:14] Iteration 26

**Container**: atlas-qwen-final Up 6 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T16:02:14.175104Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 824 tokens (stop) 38.5 tok/s, TTFT=926.5ms
- [2m2026-05-25T16:02:14.206013Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=30, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:02:32.770096Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 662 tokens (stop) 39.3 tok/s, TTFT=1694.4ms
- [2m2026-05-25T16:02:32.819203Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=32, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:03:16.442668Z[0m [33m WARN[0m [2mspark::api::chat_stream::tool_handlers[0m[2m:[0m tool call validation error (stream Δ): Error: Unknown tool 'bash command'. Available tools: bash, edit, glob, grep, read, skill, task, webfetch, write; replacing with content and ending [3mtool[0m[2m=[0mbash command

**Active opencode runs**: 3 processes
  PID 2731168 11:58 ?
  PID 2731750 11:58 ?
  PID 2731752 11:58 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779725054,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 3
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [12:05:44] Iteration 27

**Container**: atlas-qwen-final Up 8 minutes
**Recent container activity** (last 2m):
- 

**Active opencode runs**: 3 processes
  PID 2731168 11:58 ?
  PID 2731750 11:58 ?
  PID 2731752 11:58 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779725144,"owned_by":"atlas-spark"

**Status**: No container activity in last 2m / opencode procs: 3
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [12:07:14] Iteration 28

**Container**: atlas-qwen-final Up 9 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T16:06:42.452067Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 8192 tokens (length) 33.1 tok/s, TTFT=1833.5ms
- [2m2026-05-25T16:06:42.530220Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=35, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:06:56.078615Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 122 tokens (stop) 30.4 tok/s, TTFT=9508.6ms
- [2m2026-05-25T16:06:56.126457Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=37, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:07:02.157042Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 132 tokens (stop) 31.5 tok/s, TTFT=1693.6ms
- [2m2026-05-25T16:07:02.196795Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=39, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:07:05.042720Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 67 tokens (stop) 32.6 tok/s, TTFT=761.3ms
- [2m2026-05-25T16:07:05.076449Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=41, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None

**Active opencode runs**: 3 processes
  PID 2731168 11:58 ?
  PID 2731750 11:58 ?
  PID 2731752 11:58 ?
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779725235,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 3
**Stalls detected**: opencode still running
**Notable**: Auto-poll iteration

---

## [12:08:45] Iteration 29

**Container**: atlas-qwen-final Up 11 minutes
**Recent container activity** (last 2m):
- [2m2026-05-25T16:07:22.190825Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=43, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:07:25.317330Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 75 tokens (stop) 32.4 tok/s, TTFT=769.2ms
- [2m2026-05-25T16:07:25.345064Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=45, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:07:28.308645Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 70 tokens (stop) 32.5 tok/s, TTFT=780.2ms
- [2m2026-05-25T16:07:28.330363Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=47, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:07:31.129116Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 70 tokens (stop) 32.2 tok/s, TTFT=601.3ms
- [2m2026-05-25T16:07:31.161280Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=49, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:07:33.296621Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 48 tokens (stop) 31.9 tok/s, TTFT=602.5ms

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779725325,"owned_by":"atlas-spark"

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [12:10:15] Iteration 30

**Container**: NO ATLAS CONTAINERS
**Recent container activity** (last 2m):
- Error response from daemon: No such container: atlas-qwen-final

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 3
**API health**: 

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [12:11:45] Iteration 31

**Container**: NO ATLAS CONTAINERS
**Recent container activity** (last 2m):
- Error response from daemon: No such container: atlas-qwen-final

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: 

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [12:13:15] Iteration 32

**Container**: NO ATLAS CONTAINERS
**Recent container activity** (last 2m):
- Error response from daemon: No such container: atlas-qwen-final

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 3
**API health**: 

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [12:14:46] Iteration 33

**Container**: NO ATLAS CONTAINERS
**Recent container activity** (last 2m):
- Error response from daemon: No such container: atlas-qwen-final

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: 

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [12:16:16] Iteration 34

**Container**: NO ATLAS CONTAINERS
**Recent container activity** (last 2m):
- Error response from daemon: No such container: atlas-qwen-final

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: 

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [12:17:46] Iteration 35

**Container**: NO ATLAS CONTAINERS
**Recent container activity** (last 2m):
- Error response from daemon: No such container: atlas-qwen-final

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: 

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

---

## [12:19:16] Iteration 36

**Container**: NO ATLAS CONTAINERS
**Recent container activity** (last 2m):
- Error response from daemon: No such container: atlas-qwen-final

**Active opencode runs**: 0 processes
  PID 
**Active builds**: 0
**API health**: 

**Status**: Active / opencode procs: 0
**Stalls detected**: none
**Notable**: Auto-poll iteration

## [12:59:13] Iteration 37 (resumed)

**Container**: atlas-qwen-final Up 3 minutes
**Recent activity** (last 2m):
- [2m2026-05-25T16:57:25.990269Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=3, tools=0, tools_active=false, tool_choice=None, stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:57:26.016531Z[0m [32m INFO[0m [2mspark::api::chat[0m[2m:[0m Request: model=Qwen/Qwen3.6-35B-A3B-FP8, messages=2, tools=9, tools_active=true, tool_choice=Some(Mode("auto")), stream=true, temp=Some(0.3), max_tokens=8192, freq_pen=None, rep_pen=None
- [2m2026-05-25T16:58:32.836663Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 849 tokens (stop) 12.9 tok/s, TTFT=1001.4ms

**Active opencode**: count=3 PID=2787126 started=12:57 PID=2787708 started=12:57 PID=2787709 started=12:57
**Active builds**: none
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779728353,"owned_by":"atlas-spark"}]}

**Status**: healthy
**Notable**: Auto-poll iteration 37

---

## [13:00:43] Iteration 38 (resumed)

**Container**: atlas-qwen-final Up 4 minutes
**Recent activity** (last 2m):
- (none)

**Active opencode**: count=3 PID=2787126 started=12:57 PID=2787708 started=12:57 PID=2787709 started=12:57
**Active builds**: none
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779728444,"owned_by":"atlas-spark"}]}

**Status**: healthy
**Notable**: Auto-poll iteration 38

---

## [13:02:14] Iteration 39 (resumed)

**Container**: atlas-qwen-final Up 6 minutes
**Recent activity** (last 2m):
- (none)

**Active opencode**: count=3 PID=2787126 started=12:57 PID=2787708 started=12:57 PID=2787709 started=12:57
**Active builds**: none
**API health**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","created":1779728534,"owned_by":"atlas-spark"}]}

**Status**: healthy
**Notable**: Auto-poll iteration 39

---

## [13:03:54] Iteration 39
**Container**: atlas-qwen-final Up 8 minutes
**Recent (2m)**:
- [2m2026-05-25T17:02:26.049470Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 7766 tokens (length) 28.1 tok/s, TTFT=22816.6ms
- [2m2026-05-25T17:03:35.509962Z[0m [32m INFO[0m [2mspark::scheduler::lifecycle[0m[2m:[0m Done: 1025 tokens (stop) 27.0 tok/s, TTFT=31389.4ms
**opencode**: count=3  2787126@12:57 2787708@12:57 2787709@12:57 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: **STALL DETECTED**: opencode running 389s with no recent container activity
---

## [13:04:54] Iteration 40
**Container**: atlas-qwen-final Up 9 minutes
**Recent (5m)**:
- 2026-05-25T17:02:26.049470Z  INFO spark::scheduler::lifecycle: Done: 7766 tokens (length) 28.1 tok/s, TTFT=22816.6ms
- 2026-05-25T17:03:35.509962Z  INFO spark::scheduler::lifecycle: Done: 1025 tokens (stop) 27.0 tok/s, TTFT=31389.4ms
- 2026-05-25T17:03:55.853976Z  INFO spark::scheduler::lifecycle: Done: 533 tokens (stop) 29.1 tok/s, TTFT=1825.7ms
- 2026-05-25T17:04:36.222218Z  INFO spark::scheduler::lifecycle: Done: 1114 tokens (stop) 29.1 tok/s, TTFT=2038.5ms
**opencode**: count=3  2787126@12:57 2787708@12:57 2787709@12:57 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:06:24] Iteration 41
**Container**: atlas-qwen-final Up 10 minutes
**Recent (5m)**:
- 2026-05-25T17:02:26.049470Z  INFO spark::scheduler::lifecycle: Done: 7766 tokens (length) 28.1 tok/s, TTFT=22816.6ms
- 2026-05-25T17:03:35.509962Z  INFO spark::scheduler::lifecycle: Done: 1025 tokens (stop) 27.0 tok/s, TTFT=31389.4ms
- 2026-05-25T17:03:55.853976Z  INFO spark::scheduler::lifecycle: Done: 533 tokens (stop) 29.1 tok/s, TTFT=1825.7ms
- 2026-05-25T17:04:36.222218Z  INFO spark::scheduler::lifecycle: Done: 1114 tokens (stop) 29.1 tok/s, TTFT=2038.5ms
- 2026-05-25T17:06:12.087658Z  INFO spark::scheduler::lifecycle: Done: 2502 tokens (length) 26.7 tok/s, TTFT=2175.9ms
- 2026-05-25T17:06:17.712727Z  INFO spark::scheduler::lifecycle: Done: 94 tokens (stop) 28.7 tok/s, TTFT=2268.1ms
**opencode**: count=3  2787126@12:57 2787708@12:57 2787709@12:57 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:07:54] Iteration 42
**Container**: atlas-qwen-final Up 12 minutes
**Recent (5m)**:
- 2026-05-25T17:06:12.087658Z  INFO spark::scheduler::lifecycle: Done: 2502 tokens (length) 26.7 tok/s, TTFT=2175.9ms
- 2026-05-25T17:06:17.712727Z  INFO spark::scheduler::lifecycle: Done: 94 tokens (stop) 28.7 tok/s, TTFT=2268.1ms
- 2026-05-25T17:06:58.036923Z  INFO spark::scheduler::lifecycle: Done: 889 tokens (stop) 26.9 tok/s, TTFT=7248.5ms
- 2026-05-25T17:07:16.100310Z  INFO spark::scheduler::lifecycle: Done: 422 tokens (stop) 26.5 tok/s, TTFT=2076.3ms
- 2026-05-25T17:07:40.242340Z  INFO spark::scheduler::lifecycle: Done: 583 tokens (stop) 26.9 tok/s, TTFT=2155.8ms
- 2026-05-25T17:07:49.395556Z  INFO spark::scheduler::lifecycle: Done: 174 tokens (stop) 26.5 tok/s, TTFT=2366.7ms
**opencode**: count=3  2787126@12:57 2787708@12:57 2787709@12:57 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:09:24] Iteration 43
**Container**: atlas-qwen-final Up 13 minutes
**Recent (5m)**:
- 2026-05-25T17:07:16.100310Z  INFO spark::scheduler::lifecycle: Done: 422 tokens (stop) 26.5 tok/s, TTFT=2076.3ms
- 2026-05-25T17:07:40.242340Z  INFO spark::scheduler::lifecycle: Done: 583 tokens (stop) 26.9 tok/s, TTFT=2155.8ms
- 2026-05-25T17:07:49.395556Z  INFO spark::scheduler::lifecycle: Done: 174 tokens (stop) 26.5 tok/s, TTFT=2366.7ms
- 2026-05-25T17:07:57.323642Z  INFO spark::scheduler::lifecycle: Done: 139 tokens (stop) 26.5 tok/s, TTFT=2611.8ms
- 2026-05-25T17:08:06.641822Z  INFO spark::scheduler::lifecycle: Done: 167 tokens (stop) 26.1 tok/s, TTFT=2845.5ms
- 2026-05-25T17:08:37.923045Z  WARN spark::api::chat_stream::handle_token: in-think tool-call leak detected; cancelling sequence (finish_reason will be "length") model=Qwen/Qwen3.6-35B-A3B-FP8 request_id=chatcmpl-9953c988-9ecb-439c-af74-0d8ff59fe0dd opener="<parameter=" tail=orrupted between thought and execution.
**opencode**: count=3  2787126@12:57 2787708@12:57 2787709@12:57 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:10:54] Iteration 44
**Container**: atlas-qwen-final Up 15 minutes
**Recent (5m)**:
- 2026-05-25T17:07:16.100310Z  INFO spark::scheduler::lifecycle: Done: 422 tokens (stop) 26.5 tok/s, TTFT=2076.3ms
- 2026-05-25T17:07:40.242340Z  INFO spark::scheduler::lifecycle: Done: 583 tokens (stop) 26.9 tok/s, TTFT=2155.8ms
- 2026-05-25T17:07:49.395556Z  INFO spark::scheduler::lifecycle: Done: 174 tokens (stop) 26.5 tok/s, TTFT=2366.7ms
- 2026-05-25T17:07:57.323642Z  INFO spark::scheduler::lifecycle: Done: 139 tokens (stop) 26.5 tok/s, TTFT=2611.8ms
- 2026-05-25T17:08:06.641822Z  INFO spark::scheduler::lifecycle: Done: 167 tokens (stop) 26.1 tok/s, TTFT=2845.5ms
- 2026-05-25T17:08:37.923045Z  WARN spark::api::chat_stream::handle_token: in-think tool-call leak detected; cancelling sequence (finish_reason will be "length") model=Qwen/Qwen3.6-35B-A3B-FP8 request_id=chatcmpl-9953c988-9ecb-439c-af74-0d8ff59fe0dd opener="<parameter=" tail=orrupted between thought and execution.
**opencode**: count=3  2787126@12:57 2787708@12:57 2787709@12:57 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:12:24] Iteration 45
**Container**: atlas-qwen-final Up 16 minutes
**Recent (5m)**:
- 2026-05-25T17:07:40.242340Z  INFO spark::scheduler::lifecycle: Done: 583 tokens (stop) 26.9 tok/s, TTFT=2155.8ms
- 2026-05-25T17:07:49.395556Z  INFO spark::scheduler::lifecycle: Done: 174 tokens (stop) 26.5 tok/s, TTFT=2366.7ms
- 2026-05-25T17:07:57.323642Z  INFO spark::scheduler::lifecycle: Done: 139 tokens (stop) 26.5 tok/s, TTFT=2611.8ms
- 2026-05-25T17:08:06.641822Z  INFO spark::scheduler::lifecycle: Done: 167 tokens (stop) 26.1 tok/s, TTFT=2845.5ms
- 2026-05-25T17:08:37.923045Z  WARN spark::api::chat_stream::handle_token: in-think tool-call leak detected; cancelling sequence (finish_reason will be "length") model=Qwen/Qwen3.6-35B-A3B-FP8 request_id=chatcmpl-9953c988-9ecb-439c-af74-0d8ff59fe0dd opener="<parameter=" tail=orrupted between thought and execution.
- 2026-05-25T17:11:38.416632Z  INFO spark::scheduler::lifecycle: Done: 5123 tokens (stop) 24.5 tok/s, TTFT=3028.1ms
**opencode**: count=0  
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:13:54] Iteration 46
**Container**: atlas-qwen-final Up 18 minutes
**Recent (5m)**:
- 2026-05-25T17:11:38.416632Z  INFO spark::scheduler::lifecycle: Done: 5123 tokens (stop) 24.5 tok/s, TTFT=3028.1ms
- 2026-05-25T17:12:57.495252Z  INFO spark::scheduler::lifecycle: Done: 355 tokens (stop) 65.3 tok/s, TTFT=352.7ms
- 2026-05-25T17:13:03.297702Z  INFO spark::scheduler::lifecycle: Done: 355 tokens (stop) 65.1 tok/s, TTFT=348.7ms
**opencode**: count=0  
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:14:42] Iteration 46
**Container**: atlas-qwen-final Up 18 minutes
**Recent (5m)**:
- 2026-05-25T17:11:38.416632Z  INFO spark::scheduler::lifecycle: Done: 5123 tokens (stop) 24.5 tok/s, TTFT=3028.1ms
- 2026-05-25T17:12:57.495252Z  INFO spark::scheduler::lifecycle: Done: 355 tokens (stop) 65.3 tok/s, TTFT=352.7ms
- 2026-05-25T17:13:03.297702Z  INFO spark::scheduler::lifecycle: Done: 355 tokens (stop) 65.1 tok/s, TTFT=348.7ms
**opencode**: count=0  
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:16:12] Iteration 47
**Container**: NO ATLAS CONTAINERS
**Recent (5m)**:
- (none)
**opencode**: count=0  
**Builds**: 2802710 /bin/bash -c source /workspace/.claude/shell-snapshots/snapshot-bash-1779583699755-64d01z.sh 2>/dev/null || true && shopt -u extglob 2>/dev/null || true && eval 'sudo docker rm -f atlas-qwen-final 2>&1 | head -1 nohup sudo docker build -f docker/gb10/Dockerfile -t atlas-gb10:fp8-noprompt . > /tmp/atlas-np-build.log 2>&1 & echo build pid: $! until grep -qE "naming to docker.io|ERROR:|error\[E[0-9]+\]|failed to solve" /tmp/atlas-np-build.log 2>/dev/null; do sleep 10; done; echo DONE tail -3 /tmp/atlas-np-build.log' < /dev/null && pwd -P >| /tmp/claude-9740-cwd
2803364 sudo docker build -f docker/gb10/Dockerfile -t atlas-gb10:fp8-noprompt .
**API**: 
**Status**: waiting
---

## [13:17:42] Iteration 48
**Container**: atlas-qwen-final Up About a minute
**Recent (5m)**:
- (none)
**opencode**: count=0  
**Builds**: none
**API**: 
**Status**: **STALL DETECTED**: Container up but API unresponsive
---

## [13:19:12] Iteration 49
**Container**: atlas-qwen-final Up 2 minutes
**Recent (5m)**:
- 2026-05-25T17:18:38.425243Z  INFO spark::scheduler::lifecycle: Done: 296 tokens (stop) 22.6 tok/s, TTFT=11278.8ms
- 2026-05-25T17:18:54.745567Z  INFO spark::scheduler::lifecycle: Done: 406 tokens (stop) 9.9 tok/s, TTFT=981.0ms
- 2026-05-25T17:18:57.283408Z  INFO spark::scheduler::lifecycle: Done: 225 tokens (length) 31.0 tok/s, TTFT=11478.7ms
**opencode**: count=3  2811514@13:18 2812096@13:18 2812098@13:18 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:20:42] Iteration 50
**Container**: atlas-qwen-final Up 4 minutes
**Recent (5m)**:
- 2026-05-25T17:18:38.425243Z  INFO spark::scheduler::lifecycle: Done: 296 tokens (stop) 22.6 tok/s, TTFT=11278.8ms
- 2026-05-25T17:18:54.745567Z  INFO spark::scheduler::lifecycle: Done: 406 tokens (stop) 9.9 tok/s, TTFT=981.0ms
- 2026-05-25T17:18:57.283408Z  INFO spark::scheduler::lifecycle: Done: 225 tokens (length) 31.0 tok/s, TTFT=11478.7ms
- 2026-05-25T17:19:18.836663Z  INFO spark::scheduler::lifecycle: Done: 415 tokens (stop) 42.8 tok/s, TTFT=11446.1ms
- 2026-05-25T17:20:19.930081Z  INFO spark::scheduler::lifecycle: Done: 2544 tokens (length) 43.1 tok/s, TTFT=2052.2ms
- 2026-05-25T17:20:29.758372Z  INFO spark::scheduler::lifecycle: Done: 334 tokens (stop) 43.9 tok/s, TTFT=2152.5ms
**opencode**: count=3  2811514@13:18 2812096@13:18 2812098@13:18 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:22:13] Iteration 51
**Container**: atlas-qwen-final Up 5 minutes
**Recent (5m)**:
- 2026-05-25T17:18:57.283408Z  INFO spark::scheduler::lifecycle: Done: 225 tokens (length) 31.0 tok/s, TTFT=11478.7ms
- 2026-05-25T17:19:18.836663Z  INFO spark::scheduler::lifecycle: Done: 415 tokens (stop) 42.8 tok/s, TTFT=11446.1ms
- 2026-05-25T17:20:19.930081Z  INFO spark::scheduler::lifecycle: Done: 2544 tokens (length) 43.1 tok/s, TTFT=2052.2ms
- 2026-05-25T17:20:29.758372Z  INFO spark::scheduler::lifecycle: Done: 334 tokens (stop) 43.9 tok/s, TTFT=2152.5ms
- 2026-05-25T17:21:21.256671Z  INFO spark::scheduler::lifecycle: Done: 2143 tokens (length) 43.6 tok/s, TTFT=2283.1ms
- 2026-05-25T17:21:25.254800Z  INFO spark::scheduler::lifecycle: Done: 67 tokens (stop) 42.3 tok/s, TTFT=2352.1ms
**opencode**: count=3  2811514@13:18 2812096@13:18 2812098@13:18 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:23:43] Iteration 52
**Container**: atlas-qwen-final Up 7 minutes
**Recent (5m)**:
- 2026-05-25T17:22:52.186490Z  INFO spark::scheduler::lifecycle: Done: 682 tokens (length) 42.7 tok/s, TTFT=2901.5ms
- 2026-05-25T17:23:04.039426Z  INFO spark::scheduler::lifecycle: Done: 385 tokens (stop) 43.7 tok/s, TTFT=2980.3ms
- 2026-05-25T17:23:09.244722Z  INFO spark::scheduler::lifecycle: Done: 85 tokens (stop) 41.9 tok/s, TTFT=3116.2ms
- 2026-05-25T17:23:18.501585Z  INFO spark::scheduler::lifecycle: Done: 232 tokens (stop) 39.4 tok/s, TTFT=3309.3ms
- 2026-05-25T17:23:27.514090Z  INFO spark::scheduler::lifecycle: Done: 229 tokens (stop) 41.6 tok/s, TTFT=3451.1ms
- 2026-05-25T17:23:36.043742Z  INFO spark::scheduler::lifecycle: Done: 197 tokens (stop) 40.4 tok/s, TTFT=3595.9ms
**opencode**: count=3  2811514@13:18 2812096@13:18 2812098@13:18 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:25:13] Iteration 53
**Container**: atlas-qwen-final Up 8 minutes
**Recent (5m)**:
- 2026-05-25T17:23:18.501585Z  INFO spark::scheduler::lifecycle: Done: 232 tokens (stop) 39.4 tok/s, TTFT=3309.3ms
- 2026-05-25T17:23:27.514090Z  INFO spark::scheduler::lifecycle: Done: 229 tokens (stop) 41.6 tok/s, TTFT=3451.1ms
- 2026-05-25T17:23:36.043742Z  INFO spark::scheduler::lifecycle: Done: 197 tokens (stop) 40.4 tok/s, TTFT=3595.9ms
- 2026-05-25T17:23:49.731906Z  INFO spark::scheduler::lifecycle: Done: 409 tokens (stop) 41.1 tok/s, TTFT=3673.7ms
- 2026-05-25T17:24:09.003970Z  INFO spark::scheduler::lifecycle: Done: 241 tokens (stop) 40.2 tok/s, TTFT=13242.0ms
- 2026-05-25T17:24:30.337606Z  INFO spark::scheduler::lifecycle: Done: 709 tokens (stop) 40.6 tok/s, TTFT=3838.1ms
**opencode**: count=0  
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:26:43] Iteration 54
**Container**: NO ATLAS CONTAINERS
**Recent (5m)**:
- (none)
**opencode**: count=0  
**Builds**: 2815869 /bin/bash -c source /workspace/.claude/shell-snapshots/snapshot-bash-1779583699755-64d01z.sh 2>/dev/null || true && shopt -u extglob 2>/dev/null || true && eval 'cd /workspace/atlas-mtp nohup sudo docker build -f docker/gb10/Dockerfile -t atlas-gb10:fp8-much-better . > /tmp/atlas-mb-build.log 2>&1 & echo build pid: $! until grep -qE "naming to docker.io|ERROR:|error\[E[0-9]+\]|failed to solve" /tmp/atlas-mb-build.log 2>/dev/null; do sleep 10; done; echo DONE tail -3 /tmp/atlas-mb-build.log' < /dev/null && pwd -P >| /tmp/claude-0548-cwd
2816450 sudo docker build -f docker/gb10/Dockerfile -t atlas-gb10:fp8-much-better .
**API**: 
**Status**: waiting
---

## [13:28:13] Iteration 55
**Container**: atlas-qwen-final Up 41 seconds
**Recent (5m)**:
- (none)
**opencode**: count=0  
**Builds**: none
**API**: 
**Status**: **STALL DETECTED**: Container up but API unresponsive
---

## [13:29:43] Iteration 56
**Container**: atlas-qwen-final Up 2 minutes
**Recent (5m)**:
- (none)
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:31:13] Iteration 57
**Container**: atlas-qwen-final Up 3 minutes
**Recent (5m)**:
- 2026-05-25T17:29:50.549401Z  INFO spark::scheduler::lifecycle: Done: 418 tokens (stop) 23.2 tok/s, TTFT=11310.3ms
- 2026-05-25T17:30:21.479170Z  INFO spark::scheduler::lifecycle: Done: 849 tokens (stop) 14.0 tok/s, TTFT=990.4ms
- 2026-05-25T17:31:03.707450Z  INFO spark::scheduler::lifecycle: Done: 110 tokens (stop) 1.8 tok/s, TTFT=11441.6ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:32:44] Iteration 58
**Container**: atlas-qwen-final Up 5 minutes
**Recent (5m)**:
- 2026-05-25T17:31:24.700521Z  INFO spark::scheduler::lifecycle: Done: 398 tokens (stop) 42.2 tok/s, TTFT=11497.4ms
- 2026-05-25T17:31:29.868039Z  INFO spark::scheduler::lifecycle: Done: 114 tokens (stop) 41.5 tok/s, TTFT=2166.0ms
- 2026-05-25T17:31:34.740617Z  INFO spark::scheduler::lifecycle: Done: 106 tokens (stop) 41.5 tok/s, TTFT=2282.8ms
- 2026-05-25T17:31:40.570836Z  INFO spark::scheduler::lifecycle: Done: 142 tokens (stop) 41.9 tok/s, TTFT=2405.8ms
- 2026-05-25T17:32:32.889695Z  INFO spark::scheduler::lifecycle: Done: 2121 tokens (length) 42.7 tok/s, TTFT=2567.1ms
- 2026-05-25T17:32:40.315914Z  INFO spark::scheduler::lifecycle: Done: 196 tokens (stop) 41.7 tok/s, TTFT=2684.4ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:34:14] Iteration 59
**Container**: atlas-qwen-final Up 6 minutes
**Recent (5m)**:
- 2026-05-25T17:32:58.430660Z  INFO spark::scheduler::lifecycle: Done: 426 tokens (stop) 42.1 tok/s, TTFT=2919.3ms
- 2026-05-25T17:33:22.370444Z  INFO spark::scheduler::lifecycle: Done: 854 tokens (stop) 41.1 tok/s, TTFT=3116.4ms
- 2026-05-25T17:33:27.835503Z  INFO spark::scheduler::lifecycle: Done: 89 tokens (stop) 41.3 tok/s, TTFT=3239.1ms
- 2026-05-25T17:33:40.793330Z  INFO spark::scheduler::lifecycle: Done: 396 tokens (stop) 41.3 tok/s, TTFT=3322.1ms
- 2026-05-25T17:34:00.498443Z  INFO spark::scheduler::lifecycle: Done: 689 tokens (stop) 42.7 tok/s, TTFT=3499.5ms
- 2026-05-25T17:34:12.047418Z  INFO spark::scheduler::lifecycle: Done: 330 tokens (stop) 42.4 tok/s, TTFT=3692.0ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:35:44] Iteration 60
**Container**: atlas-qwen-final Up 8 minutes
**Recent (5m)**:
- 2026-05-25T17:34:41.254248Z  INFO spark::scheduler::lifecycle: Done: 480 tokens (stop) 40.5 tok/s, TTFT=4131.1ms
- 2026-05-25T17:34:58.166068Z  INFO spark::scheduler::lifecycle: Done: 124 tokens (stop) 40.2 tok/s, TTFT=13763.5ms
- 2026-05-25T17:35:09.072393Z  INFO spark::scheduler::lifecycle: Done: 258 tokens (stop) 39.9 tok/s, TTFT=4376.7ms
- 2026-05-25T17:35:18.408448Z  INFO spark::scheduler::lifecycle: Done: 173 tokens (stop) 39.2 tok/s, TTFT=4858.7ms
- 2026-05-25T17:35:27.780106Z  INFO spark::scheduler::lifecycle: Done: 139 tokens (stop) 39.0 tok/s, TTFT=5718.4ms
- 2026-05-25T17:35:35.216580Z  INFO spark::scheduler::lifecycle: Done: 250 tokens (stop) 38.6 tok/s, TTFT=905.6ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:37:14] Iteration 61
**Container**: atlas-qwen-final Up 9 minutes
**Recent (5m)**:
- 2026-05-25T17:35:35.216580Z  INFO spark::scheduler::lifecycle: Done: 250 tokens (stop) 38.6 tok/s, TTFT=905.6ms
- 2026-05-25T17:35:58.535091Z  INFO spark::scheduler::lifecycle: Done: 864 tokens (stop) 38.9 tok/s, TTFT=1030.4ms
- 2026-05-25T17:36:17.566135Z  INFO spark::scheduler::lifecycle: Done: 124 tokens (stop) 38.6 tok/s, TTFT=15632.4ms
- 2026-05-25T17:36:36.755275Z  INFO spark::scheduler::lifecycle: Done: 701 tokens (stop) 39.3 tok/s, TTFT=1305.5ms
- 2026-05-25T17:36:41.486555Z  INFO spark::scheduler::lifecycle: Done: 118 tokens (stop) 36.6 tok/s, TTFT=1434.7ms
- 2026-05-25T17:37:05.419721Z  INFO spark::scheduler::lifecycle: Done: 869 tokens (length) 39.1 tok/s, TTFT=1592.3ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:38:44] Iteration 62
**Container**: atlas-qwen-final Up 11 minutes
**Recent (5m)**:
- 2026-05-25T17:37:36.808775Z  INFO spark::scheduler::lifecycle: Done: 415 tokens (stop) 38.2 tok/s, TTFT=1795.0ms
- 2026-05-25T17:37:42.266187Z  INFO spark::scheduler::lifecycle: Done: 131 tokens (stop) 37.8 tok/s, TTFT=1937.1ms
- 2026-05-25T17:37:47.841743Z  INFO spark::scheduler::lifecycle: Done: 131 tokens (stop) 38.0 tok/s, TTFT=2058.5ms
- 2026-05-25T17:38:07.216449Z  INFO spark::scheduler::lifecycle: Done: 102 tokens (stop) 38.2 tok/s, TTFT=16651.0ms
- 2026-05-25T17:38:13.225814Z  INFO spark::scheduler::lifecycle: Done: 139 tokens (stop) 37.8 tok/s, TTFT=2276.4ms
- 2026-05-25T17:38:39.954170Z  INFO spark::scheduler::lifecycle: Done: 897 tokens (stop) 37.2 tok/s, TTFT=2589.2ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:40:14] Iteration 63
**Container**: atlas-qwen-final Up 12 minutes
**Recent (5m)**:
- 2026-05-25T17:37:42.266187Z  INFO spark::scheduler::lifecycle: Done: 131 tokens (stop) 37.8 tok/s, TTFT=1937.1ms
- 2026-05-25T17:37:47.841743Z  INFO spark::scheduler::lifecycle: Done: 131 tokens (stop) 38.0 tok/s, TTFT=2058.5ms
- 2026-05-25T17:38:07.216449Z  INFO spark::scheduler::lifecycle: Done: 102 tokens (stop) 38.2 tok/s, TTFT=16651.0ms
- 2026-05-25T17:38:13.225814Z  INFO spark::scheduler::lifecycle: Done: 139 tokens (stop) 37.8 tok/s, TTFT=2276.4ms
- 2026-05-25T17:38:39.954170Z  INFO spark::scheduler::lifecycle: Done: 897 tokens (stop) 37.2 tok/s, TTFT=2589.2ms
- 2026-05-25T17:39:52.349569Z  INFO spark::scheduler::lifecycle: Done: 2672 tokens (length) 38.4 tok/s, TTFT=2684.9ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:41:45] Iteration 64
**Container**: atlas-qwen-final Up 14 minutes
**Recent (5m)**:
- 2026-05-25T17:39:52.349569Z  INFO spark::scheduler::lifecycle: Done: 2672 tokens (length) 38.4 tok/s, TTFT=2684.9ms
- 2026-05-25T17:40:32.022043Z  INFO spark::scheduler::lifecycle: Done: 838 tokens (stop) 37.4 tok/s, TTFT=17202.4ms
- 2026-05-25T17:40:50.763568Z  INFO spark::scheduler::lifecycle: Done: 624 tokens (stop) 39.4 tok/s, TTFT=2833.5ms
- 2026-05-25T17:40:56.260580Z  INFO spark::scheduler::lifecycle: Done: 93 tokens (stop) 37.6 tok/s, TTFT=2960.9ms
- 2026-05-25T17:41:16.593748Z  INFO spark::scheduler::lifecycle: Done: 271 tokens (stop) 36.4 tok/s, TTFT=12814.2ms
- 2026-05-25T17:41:41.859474Z  INFO spark::scheduler::lifecycle: Done: 813 tokens (stop) 37.1 tok/s, TTFT=3282.6ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:43:15] Iteration 65
**Container**: atlas-qwen-final Up 15 minutes
**Recent (5m)**:
- 2026-05-25T17:41:16.593748Z  INFO spark::scheduler::lifecycle: Done: 271 tokens (stop) 36.4 tok/s, TTFT=12814.2ms
- 2026-05-25T17:41:41.859474Z  INFO spark::scheduler::lifecycle: Done: 813 tokens (stop) 37.1 tok/s, TTFT=3282.6ms
- 2026-05-25T17:42:00.200465Z  INFO spark::scheduler::lifecycle: Done: 556 tokens (stop) 37.4 tok/s, TTFT=3410.9ms
- 2026-05-25T17:42:23.237994Z  INFO spark::scheduler::lifecycle: Done: 176 tokens (stop) 36.8 tok/s, TTFT=18182.7ms
- 2026-05-25T17:42:31.862958Z  INFO spark::scheduler::lifecycle: Done: 169 tokens (stop) 35.8 tok/s, TTFT=3836.7ms
- 2026-05-25T17:42:55.948777Z  INFO spark::scheduler::lifecycle: Done: 738 tokens (stop) 36.9 tok/s, TTFT=4029.7ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:44:45] Iteration 66
**Container**: atlas-qwen-final Up 17 minutes
**Recent (5m)**:
- 2026-05-25T17:42:55.948777Z  INFO spark::scheduler::lifecycle: Done: 738 tokens (stop) 36.9 tok/s, TTFT=4029.7ms
- 2026-05-25T17:43:16.031946Z  INFO spark::scheduler::lifecycle: Done: 212 tokens (stop) 35.8 tok/s, TTFT=14088.6ms
- 2026-05-25T17:43:24.391767Z  INFO spark::scheduler::lifecycle: Done: 139 tokens (stop) 36.4 tok/s, TTFT=4481.5ms
- 2026-05-25T17:43:37.744081Z  INFO spark::scheduler::lifecycle: Done: 204 tokens (stop) 23.5 tok/s, TTFT=4611.7ms
- 2026-05-25T17:44:06.468477Z  INFO spark::scheduler::lifecycle: Done: 354 tokens (stop) 37.3 tok/s, TTFT=19167.9ms
- 2026-05-25T17:44:24.791870Z  INFO spark::scheduler::lifecycle: Done: 471 tokens (stop) 35.2 tok/s, TTFT=4885.5ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:46:15] Iteration 67
**Container**: atlas-qwen-final Up 18 minutes
**Recent (5m)**:
- 2026-05-25T17:43:37.744081Z  INFO spark::scheduler::lifecycle: Done: 204 tokens (stop) 23.5 tok/s, TTFT=4611.7ms
- 2026-05-25T17:44:06.468477Z  INFO spark::scheduler::lifecycle: Done: 354 tokens (stop) 37.3 tok/s, TTFT=19167.9ms
- 2026-05-25T17:44:24.791870Z  INFO spark::scheduler::lifecycle: Done: 471 tokens (stop) 35.2 tok/s, TTFT=4885.5ms
- 2026-05-25T17:45:06.559677Z  INFO spark::scheduler::lifecycle: Done: 1147 tokens (stop) 36.1 tok/s, TTFT=9910.4ms
- 2026-05-25T17:45:35.771807Z  INFO spark::scheduler::lifecycle: Done: 853 tokens (stop) 35.6 tok/s, TTFT=5216.3ms
- 2026-05-25T17:45:51.242031Z  INFO spark::scheduler::lifecycle: Done: 533 tokens (stop) 35.8 tok/s, TTFT=518.8ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:47:45] Iteration 68
**Container**: atlas-qwen-final Up 20 minutes
**Recent (5m)**:
- 2026-05-25T17:44:24.791870Z  INFO spark::scheduler::lifecycle: Done: 471 tokens (stop) 35.2 tok/s, TTFT=4885.5ms
- 2026-05-25T17:45:06.559677Z  INFO spark::scheduler::lifecycle: Done: 1147 tokens (stop) 36.1 tok/s, TTFT=9910.4ms
- 2026-05-25T17:45:35.771807Z  INFO spark::scheduler::lifecycle: Done: 853 tokens (stop) 35.6 tok/s, TTFT=5216.3ms
- 2026-05-25T17:45:51.242031Z  INFO spark::scheduler::lifecycle: Done: 533 tokens (stop) 35.8 tok/s, TTFT=518.8ms
- 2026-05-25T17:47:17.211253Z  INFO spark::scheduler::lifecycle: Done: 2391 tokens (length) 36.4 tok/s, TTFT=20212.3ms
- 2026-05-25T17:47:41.703499Z  INFO spark::scheduler::lifecycle: Done: 874 tokens (stop) 37.0 tok/s, TTFT=777.8ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:49:15] Iteration 69
**Container**: atlas-qwen-final Up 21 minutes
**Recent (5m)**:
- 2026-05-25T17:45:06.559677Z  INFO spark::scheduler::lifecycle: Done: 1147 tokens (stop) 36.1 tok/s, TTFT=9910.4ms
- 2026-05-25T17:45:35.771807Z  INFO spark::scheduler::lifecycle: Done: 853 tokens (stop) 35.6 tok/s, TTFT=5216.3ms
- 2026-05-25T17:45:51.242031Z  INFO spark::scheduler::lifecycle: Done: 533 tokens (stop) 35.8 tok/s, TTFT=518.8ms
- 2026-05-25T17:47:17.211253Z  INFO spark::scheduler::lifecycle: Done: 2391 tokens (length) 36.4 tok/s, TTFT=20212.3ms
- 2026-05-25T17:47:41.703499Z  INFO spark::scheduler::lifecycle: Done: 874 tokens (stop) 37.0 tok/s, TTFT=777.8ms
- 2026-05-25T17:48:46.893682Z  INFO spark::scheduler::lifecycle: Done: 2332 tokens (length) 36.3 tok/s, TTFT=925.6ms
**opencode**: count=3  2824609@13:29 2825191@13:29 2825193@13:29 
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:50:45] Iteration 70
**Container**: atlas-qwen-final Up 23 minutes
**Recent (5m)**:
- 2026-05-25T17:47:17.211253Z  INFO spark::scheduler::lifecycle: Done: 2391 tokens (length) 36.4 tok/s, TTFT=20212.3ms
- 2026-05-25T17:47:41.703499Z  INFO spark::scheduler::lifecycle: Done: 874 tokens (stop) 37.0 tok/s, TTFT=777.8ms
- 2026-05-25T17:48:46.893682Z  INFO spark::scheduler::lifecycle: Done: 2332 tokens (length) 36.3 tok/s, TTFT=925.6ms
- 2026-05-25T17:49:45.335856Z  INFO spark::scheduler::lifecycle: Done: 2087 tokens (length) 36.4 tok/s, TTFT=987.1ms
- 2026-05-25T17:49:49.168272Z  INFO spark::scheduler::lifecycle: Done: 96 tokens (stop) 35.5 tok/s, TTFT=1077.4ms
- 2026-05-25T17:50:09.930830Z  INFO spark::scheduler::lifecycle: Done: 2 tokens (length) 64.0 tok/s, TTFT=20669.6ms
**opencode**: count=0  
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:52:16] Iteration 71
**Container**: atlas-qwen-final Up 24 minutes
**Recent (5m)**:
- 2026-05-25T17:47:17.211253Z  INFO spark::scheduler::lifecycle: Done: 2391 tokens (length) 36.4 tok/s, TTFT=20212.3ms
- 2026-05-25T17:47:41.703499Z  INFO spark::scheduler::lifecycle: Done: 874 tokens (stop) 37.0 tok/s, TTFT=777.8ms
- 2026-05-25T17:48:46.893682Z  INFO spark::scheduler::lifecycle: Done: 2332 tokens (length) 36.3 tok/s, TTFT=925.6ms
- 2026-05-25T17:49:45.335856Z  INFO spark::scheduler::lifecycle: Done: 2087 tokens (length) 36.4 tok/s, TTFT=987.1ms
- 2026-05-25T17:49:49.168272Z  INFO spark::scheduler::lifecycle: Done: 96 tokens (stop) 35.5 tok/s, TTFT=1077.4ms
- 2026-05-25T17:50:09.930830Z  INFO spark::scheduler::lifecycle: Done: 2 tokens (length) 64.0 tok/s, TTFT=20669.6ms
**opencode**: count=0  
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:53:46] Iteration 72
**Container**: atlas-qwen-final Up 26 minutes
**Recent (5m)**:
- 2026-05-25T17:48:46.893682Z  INFO spark::scheduler::lifecycle: Done: 2332 tokens (length) 36.3 tok/s, TTFT=925.6ms
- 2026-05-25T17:49:45.335856Z  INFO spark::scheduler::lifecycle: Done: 2087 tokens (length) 36.4 tok/s, TTFT=987.1ms
- 2026-05-25T17:49:49.168272Z  INFO spark::scheduler::lifecycle: Done: 96 tokens (stop) 35.5 tok/s, TTFT=1077.4ms
- 2026-05-25T17:50:09.930830Z  INFO spark::scheduler::lifecycle: Done: 2 tokens (length) 64.0 tok/s, TTFT=20669.6ms
**opencode**: count=0  
**Builds**: none
**API**: {"object":"list","data":[{"id":"Qwen/Qwen3.6-35B-A3B-FP8","object":"model","crea
**Status**: healthy
---

## [13:55:16] Iteration 73
**Container**: atlas-qwen-final Up 44 seconds
**Recent (5m)**:
- (none)
**opencode**: count=0  
**Builds**: none
**API**: 
**Status**: **STALL DETECTED**: Container up but API unresponsive
---
