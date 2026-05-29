// coi-serviceworker.js
// Ensures the page is cross-origin-isolated so SharedArrayBuffer is available
// for the AudioWorklet ring buffer, even when some browsers don't fully honour
// the server's COOP/COEP headers on cached or first-visit responses.
//
// Strategy: intercept every fetch and re-stamp COOP + COEP + CORP headers onto
// the response before the browser sees it.  After registration the page reloads
// once; from then on every response already carries the headers.
//
// Based on the coi-serviceworker pattern by Guido Zuidhof (MIT licence).

self.addEventListener('install',  () => self.skipWaiting());
self.addEventListener('activate', e  => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', e => {
  // Skip opaque "only-if-cached" cross-origin requests — they can't be rewrapped.
  if (e.request.cache === 'only-if-cached' && e.request.mode !== 'same-origin') return;

  e.respondWith(
    fetch(e.request)
      .then(r => {
        if (r.status === 0) return r;   // opaque response — pass through
        const h = new Headers(r.headers);
        h.set('Cross-Origin-Opener-Policy',   'same-origin');
        h.set('Cross-Origin-Embedder-Policy', 'require-corp');
        h.set('Cross-Origin-Resource-Policy', 'same-origin');
        return new Response(r.body, { status: r.status, statusText: r.statusText, headers: h });
      })
      .catch(() => fetch(e.request))    // network failure — fall back to unmodified
  );
});
