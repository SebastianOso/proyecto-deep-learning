"use strict";

// Inversión, recorte por tinta >35, Lanczos de 3 lóbulos y centrado, como
// preparar_dibujo en Python. Se conserva uint8 hasta normalizar para la CNN.
(function (root) {
  function redondearPar(n) {
    const entero = Math.floor(n);
    return n - entero === 0.5 ? entero + entero % 2 : Math.round(n);
  }

  function coeficientes(origen, destino) {
    const escala = origen / destino;
    const filtro = Math.max(1, escala);
    const sinc = x => x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    return Array.from({ length: destino }, (_, i) => {
      const centro = (i + 0.5) * escala;
      const inicio = Math.max(0, Math.trunc(centro - 3 * filtro + 0.5));
      const fin = Math.min(origen, Math.trunc(centro + 3 * filtro + 0.5));
      const valores = [];
      for (let j = inicio; j < fin; j++) {
        const x = (j + 0.5 - centro) / filtro;
        valores.push(Math.abs(x) < 3 ? sinc(x) * sinc(x / 3) : 0);
      }
      const suma = valores.reduce((a, b) => a + b, 0);
      // Coeficientes de precisión fija y saturación uint8 entre ambas pasadas.
      return { inicio, pesos: valores.map(v => {
        const valor = v / suma * 4194304;
        return Math.trunc(valor + (valor < 0 ? -0.5 : 0.5));
      }) };
    });
  }

  function redimensionar(datos, ancho, alto, nuevoAncho, nuevoAlto) {
    const horizontal = new Uint8Array(nuevoAncho * alto);
    const salida = new Uint8Array(nuevoAncho * nuevoAlto);
    const limitar = n => Math.max(0, Math.min(255, Math.floor(n / 4194304)));
    const cx = coeficientes(ancho, nuevoAncho);
    const cy = coeficientes(alto, nuevoAlto);
    for (let y = 0; y < alto; y++) {
      for (let x = 0; x < nuevoAncho; x++) {
        const c = cx[x];
        let suma = 2097152;
        c.pesos.forEach((p, j) => { suma += p * datos[y * ancho + c.inicio + j]; });
        horizontal[y * nuevoAncho + x] = limitar(suma);
      }
    }
    for (let y = 0; y < nuevoAlto; y++) {
      for (let x = 0; x < nuevoAncho; x++) {
        const c = cy[y];
        let suma = 2097152;
        c.pesos.forEach((p, j) => { suma += p * horizontal[(c.inicio + j) * nuevoAncho + x]; });
        salida[y * nuevoAncho + x] = limitar(suma);
      }
    }
    return salida;
  }

  function prepararImagen({ data, width, height }) {
    const tinta = new Uint8Array(width * height);
    let minX = width, minY = height, maxX = -1, maxY = -1;
    let pixeles = 0, minimo = 255, maximo = 0;
    for (let i = 0; i < tinta.length; i++) {
      const gris = Math.round((299 * data[4 * i] + 587 * data[4 * i + 1] + 114 * data[4 * i + 2]) / 1000);
      tinta[i] = 255 - gris;
      minimo = Math.min(minimo, gris);
      maximo = Math.max(maximo, gris);
      if (tinta[i] > 35) {
        pixeles++;
        const x = i % width, y = Math.floor(i / width);
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
    }
    if (pixeles < 3 || maximo - minimo < 15) {
      throw new Error("El dibujo está vacío o no tiene contraste suficiente.");
    }
    const ancho = maxX - minX + 1, alto = maxY - minY + 1;
    const recorte = new Uint8Array(ancho * alto);
    for (let y = 0; y < alto; y++) {
      recorte.set(tinta.subarray((minY + y) * width + minX,
        (minY + y) * width + minX + ancho), y * ancho);
    }
    const escala = 24 / Math.max(ancho, alto);
    const nx = Math.max(1, redondearPar(ancho * escala));
    const ny = Math.max(1, redondearPar(alto * escala));
    const reducido = redimensionar(recorte, ancho, alto, nx, ny);
    const salida = new Uint8Array(28 * 28);
    const dx = Math.floor((28 - nx) / 2), dy = Math.floor((28 - ny) / 2);
    for (let y = 0; y < ny; y++) {
      salida.set(reducido.subarray(y * nx, (y + 1) * nx), (dy + y) * 28 + dx);
    }
    return salida;
  }

  root.QuickDrawPreprocesamiento = { prepararImagen };
  if (typeof module !== "undefined" && module.exports) module.exports = { prepararImagen };
})(typeof window === "undefined" ? globalThis : window);
