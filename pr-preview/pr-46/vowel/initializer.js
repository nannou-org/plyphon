// A trunk initializer that drives the loading screen: it advances the progress bar as the wasm
// downloads and removes the loading overlay once the app is running. Referenced from each example
// page via `data-initializer="initializer.js"`.
export default function initializer() {
  const loading = document.getElementById("loading");
  const bar = document.getElementById("loading-bar");
  const done = () => {
    loading?.remove();
    document.body.classList.remove("loading");
  };
  return {
    onStart: () => {},
    onProgress: ({ current, total }) => {
      if (bar && total > 0) {
        bar.style.width = `${Math.round((current / total) * 100)}%`;
      }
    },
    onComplete: done,
    onSuccess: (_wasm) => {},
    onFailure: (error) => {
      if (loading) {
        loading.innerHTML = "<p>failed to load - see the console for details</p>";
      }
      throw error;
    },
  };
}
