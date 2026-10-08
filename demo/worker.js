// worker.js — runs the Python inference in a Web Worker with Pyodide, so the
// page stays responsive.  Messages:
//   in:  {type: "run", runId, text, params}
//   out: {type: "status", message} | {type: "ready"}
//        | {type: "result", runId, result} | {type: "done", runId}
//        | {type: "error", runId, message}

const PYODIDE_VERSION = "0.27.8";
importScripts(`https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.js`);

const status = (message) => postMessage({ type: "status", message });

async function setup() {
  status("Loading the Python runtime…");
  const pyodide = await loadPyodide();

  status("Loading numpy, pandas and scipy…");
  await pyodide.loadPackage(["numpy", "pandas", "scipy", "more-itertools"]);

  status("Loading the model code…");
  const files = await (await fetch("files.json")).json();
  for (const path of files) {
    const text = await (await fetch(path)).text();
    const dir = "/app/" + path.split("/").slice(0, -1).join("/");
    pyodide.FS.mkdirTree(dir);
    pyodide.FS.writeFile("/app/" + path, text);
  }
  pyodide.runPython(`
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/app")
from inference import iter_infer, parse_segments, result_to_json
`);
  return pyodide;
}

const ready = setup().then(
  (pyodide) => { postMessage({ type: "ready" }); return pyodide; },
  (err) => { postMessage({ type: "error", message: String(err.message || err) }); throw err; }
);

onmessage = async ({ data }) => {
  if (data.type !== "run") return;
  let pyodide;
  try {
    pyodide = await ready;
  } catch {
    return;
  }
  try {
    pyodide.globals.set("melody_text", data.text);
    pyodide.globals.set("params", pyodide.toPy(data.params));
    pyodide.runPython(`
segments = parse_segments(melody_text, num_notes=params.get("num_notes", 6))
results = iter_infer(segments, **params)
`);
    const total = pyodide.runPython("len(segments)");
    for (let n = 1; n <= total; n++) {
      status(`Compressing segment ${n} of ${total}…`);
      const json = pyodide.runPython("result_to_json(next(results))");
      postMessage({ type: "result", runId: data.runId, result: JSON.parse(json) });
    }
    postMessage({ type: "done", runId: data.runId });
  } catch (err) {
    // Show only the last line of a Python traceback
    const lines = String(err.message || err).trim().split("\n");
    const message = lines[lines.length - 1].replace(/^ValueError: /, "");
    postMessage({ type: "error", runId: data.runId, message });
  }
};
