"use strict";

const canvas = document.querySelector("#lienzo");
const ctx = canvas.getContext("2d");
const predictButton = document.querySelector("#predecir");
const clearButton = document.querySelector("#limpiar");
const resultPanel = document.querySelector(".result-panel");
const result = document.querySelector("#resultado");
const emptyState = document.querySelector("#espera");
const status = document.querySelector("#estado");
let activePointer = null;
let previous = null;
let revision = 0;
let drawingExists = false;
let pending = false;

function message(text = "", error = false) {
  status.textContent = text;
  status.classList.toggle("error", error);
}

function invalidate() {
  revision += 1;
  result.hidden = true;
  emptyState.hidden = false;
  document.querySelector("#predicciones").replaceChildren();
  document.querySelector("#clase-principal").textContent = "";
  message();
}

function clear() {
  if (activePointer !== null && canvas.hasPointerCapture(activePointer)) {
    canvas.releasePointerCapture(activePointer);
  }
  activePointer = null;
  previous = null;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawingExists = false;
  invalidate();
}

function point(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * canvas.width / bounds.width,
    y: (event.clientY - bounds.top) * canvas.height / bounds.height,
  };
}

canvas.addEventListener("pointerdown", (event) => {
  if (!event.isPrimary || event.button !== 0 || activePointer !== null) return;
  event.preventDefault();
  activePointer = event.pointerId;
  canvas.setPointerCapture(event.pointerId);
  previous = point(event);
  invalidate();
  drawingExists = true;
  ctx.fillStyle = "#111111";
  ctx.beginPath();
  ctx.arc(previous.x, previous.y, 3, 0, 2 * Math.PI);
  ctx.fill();
});

canvas.addEventListener("pointermove", (event) => {
  if (event.pointerId !== activePointer) return;
  event.preventDefault();
  const current = point(event);
  ctx.strokeStyle = "#111111";
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(previous.x, previous.y);
  ctx.lineTo(current.x, current.y);
  ctx.stroke();
  previous = current;
  invalidate();
});

function finish(event) {
  if (event.pointerId !== activePointer) return;
  activePointer = null;
  previous = null;
}
canvas.addEventListener("pointerup", finish);
canvas.addEventListener("pointercancel", finish);
canvas.addEventListener("lostpointercapture", finish);
clearButton.addEventListener("click", clear);

predictButton.addEventListener("click", async () => {
  if (pending) return;
  if (!drawingExists) {
    message("Haz un dibujo antes de predecir.", true);
    return;
  }
  const requestedRevision = revision;
  const image = canvas.toDataURL("image/png");
  pending = true;
  predictButton.disabled = true;
  predictButton.textContent = "Pensando…";
  resultPanel.setAttribute("aria-busy", "true");
  message("Mirando tu dibujo…");
  try {
    const response = await fetch("/predecir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imagen: image }),
    });
    const data = await response.json();
    // A result for an older drawing must never replace the current state.
    if (revision !== requestedRevision) return;
    if (!response.ok) throw new Error(data.error || "No se pudo predecir el dibujo.");
    const predictions = document.querySelector("#predicciones");
    predictions.replaceChildren();
    const format = new Intl.NumberFormat("es-MX", { style: "percent", maximumFractionDigits: 1 });
    for (const item of data.predicciones) {
      const row = document.createElement("li");
      const heading = document.createElement("div");
      heading.className = "score-heading";
      const label = document.createElement("span");
      label.textContent = item.clase;
      const score = document.createElement("span");
      score.textContent = format.format(item.probabilidad);
      heading.append(label, score);
      const track = document.createElement("div");
      track.className = "track";
      track.setAttribute("aria-hidden", "true");
      const fill = document.createElement("div");
      fill.className = "fill";
      fill.style.width = `${item.probabilidad * 100}%`;
      track.append(fill);
      row.append(heading, track);
      predictions.append(row);
    }
    const first = data.predicciones[0];
    document.querySelector("#clase-principal").textContent = first.clase;
    emptyState.hidden = true;
    result.hidden = false;
    message(`Primera opción: ${first.clase}, ${format.format(first.probabilidad)}.`);
  } catch (error) {
    if (revision !== requestedRevision) return;
    message(error instanceof TypeError
      ? "No se pudo conectar con el servidor. Comprueba que la demo siga encendida."
      : error.message, true);
  } finally {
    pending = false;
    predictButton.disabled = false;
    predictButton.textContent = "Predecir";
    resultPanel.setAttribute("aria-busy", "false");
  }
});

clear();
