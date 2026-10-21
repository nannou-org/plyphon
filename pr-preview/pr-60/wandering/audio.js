// Browsers start every AudioContext suspended until a user gesture, and cpal does not re-resume its
// context on a later click. Proxy the AudioContext constructor so we can track each context cpal
// creates and resume them all on the first user interaction.
// https://developer.chrome.com/blog/web-audio-autoplay/#moving-forward
//
// Loaded as a classic <script> in <head> (before trunk's deferred wasm module), so the proxy is
// installed before cpal ever constructs a context.
(function () {
  const contexts = [];
  const events = [
    "click", "auxclick", "dblclick", "mousedown", "mouseup",
    "pointerup", "touchend", "keydown", "keyup",
  ];
  const Original = self.AudioContext;
  if (!Original) return;
  self.AudioContext = new Proxy(Original, {
    construct(target, args) {
      const ctx = new target(...args);
      contexts.push(ctx);
      return ctx;
    },
  });
  function resumeAll() {
    let running = 0;
    contexts.forEach((ctx) => {
      if (ctx.state !== "running") ctx.resume();
      else running++;
    });
    if (contexts.length > 0 && running === contexts.length) {
      events.forEach((e) => document.removeEventListener(e, resumeAll));
    }
  }
  events.forEach((e) => document.addEventListener(e, resumeAll));
})();
