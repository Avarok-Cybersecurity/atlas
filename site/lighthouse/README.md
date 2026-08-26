# Lighthouse gate

`index.html` must score **100 on all four categories**. That is the marketing
page and there is no excuse for anything less.

`control.html` must score 100 on performance, accessibility and SEO, and is held
to 90 on best-practices with `errors-in-console` switched off. That single
exception is deliberate and worth stating rather than burying:

The control plane's job is to find an agent on `ws://127.0.0.1:34333`. For any
visitor without one — which is nearly all of them, and always the CI runner —
that connection is refused, and Chrome writes

    WebSocket connection to 'ws://127.0.0.1:34333/ws' failed:
    Error in connection establishment: net::ERR_CONNECTION_REFUSED

to the console itself. No application code produces it and none can suppress
it: it is emitted by the network stack before any JavaScript sees the failure.
Lighthouse counts it under `errors-in-console`.

The alternatives were worse. Probing only after a click would remove the
behaviour the page exists for — it advances on its own when you start an agent
in a terminal. Probing once rather than with backoff would still log one error
and still fail the audit. So the audit is disabled for this one URL, and the
rest of best-practices stays enforced at 90 so a genuine regression still trips
the gate.

The home page keeps its 100 because its agent probe only runs when the user
clicks Run.
