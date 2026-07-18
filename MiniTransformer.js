// ==========================================
// MiniTransformer.js  (v2 — treino REAL)
// Modelo de linguagem baseado em Transformer
// (decodificador simplificado, nível de caractere)
//
// v2: agora com backpropagation completa através
// de atenção, feed-forward, layer norm e embeddings.
// A v1 só treinava a camada de saída — não aprendia
// padrões de linguagem de verdade. Esta versão aprende.
// Feito para rodar offline em navegador de celular.
// ==========================================

class MiniTransformer {
  constructor(vocabSize, dModel = 64, numHeads = 4, numLayers = 2, blockSize = 32) {
    this.vocabSize = vocabSize;
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.numLayers = numLayers;
    this.blockSize = blockSize;
    this.dFF = dModel * 4;
    this.vocabChars = null; // preenchido via definirVocabulario(), salvo/carregado junto do modelo

    this.tokenEmbedding = this._randomMatrix(vocabSize, dModel);
    this.positionEmbedding = this._randomMatrix(blockSize, dModel);

    this.layers = [];
    for (let i = 0; i < numLayers; i++) {
      this.layers.push({
        wQ: this._randomMatrix(dModel, dModel),
        wK: this._randomMatrix(dModel, dModel),
        wV: this._randomMatrix(dModel, dModel),
        wO: this._randomMatrix(dModel, dModel),
        w1: this._randomMatrix(dModel, this.dFF),
        b1: this._zeros(this.dFF),
        w2: this._randomMatrix(this.dFF, dModel),
        b2: this._zeros(dModel),
        gamma1: this._ones(dModel),
        beta1: this._zeros(dModel),
        gamma2: this._ones(dModel),
        beta2: this._zeros(dModel),
      });
    }

    this.outputWeight = this._randomMatrix(dModel, vocabSize);
    this.outputBias = this._zeros(vocabSize);
  }

  definirVocabulario(chars) {
    this.vocabChars = chars;
  }

  // ==================== HELPERS BÁSICOS ====================
  _randomMatrix(rows, cols) {
    const scale = Math.sqrt(2.0 / (rows + cols));
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
    );
  }
  _zeros(size) { return Array(size).fill(0); }
  _ones(size) { return Array(size).fill(1); }
  _zerosMatrix(rows, cols) { return Array.from({ length: rows }, () => Array(cols).fill(0)); }
  _sumRows(matrix) {
    const cols = matrix[0].length;
    const out = Array(cols).fill(0);
    for (const row of matrix) for (let j = 0; j < cols; j++) out[j] += row[j];
    return out;
  }
  _softmax(logits) {
    const max = Math.max(...logits);
    const exp = logits.map(v => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / sum);
  }
  _matmul(a, b) {
    const m = a.length, n = a[0].length, p = b[0].length;
    const result = Array.from({ length: m }, () => Array(p).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < p; j++)
        for (let k = 0; k < n; k++)
          result[i][j] += a[i][k] * b[k][j];
    return result;
  }
  _transpose(matrix) { return matrix[0].map((_, col) => matrix.map(row => row[col])); }
  _relu(x) { return Math.max(0, x); }
  _clip(v, c = 5) { return Math.max(-c, Math.min(c, v)); }

  // Layer norm com cache pra backward (mean/variance/xhat)
  _layerNormForward(x, gamma, beta, eps = 1e-5) {
    const n = x.length;
    const mean = x.reduce((a, b) => a + b, 0) / n;
    const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance + eps);
    const xhat = x.map(v => (v - mean) / std);
    const y = xhat.map((v, i) => gamma[i] * v + beta[i]);
    return { y, mean, variance };
  }

  _layerNormBackward(x, gamma, mean, variance, dy, eps = 1e-5) {
    const n = x.length;
    const std = Math.sqrt(variance + eps);
    const xhat = x.map(v => (v - mean) / std);
    const dxhat = dy.map((v, i) => v * gamma[i]);
    const dgamma = dy.map((v, i) => v * xhat[i]);
    const dbeta = dy.slice();
    const sumDxhat = dxhat.reduce((a, b) => a + b, 0);
    const sumDxhatXhat = dxhat.reduce((a, b, i) => a + b * xhat[i], 0);
    const dx = dxhat.map((v, i) => (1 / (n * std)) * (n * v - sumDxhat - xhat[i] * sumDxhatXhat));
    return { dx, dgamma, dbeta };
  }

  // Backward genérica de uma camada linear y = x·W (por posição), retorna dInput e dWeight
  _linearBackward(input, weight, dOutput) {
    const seqLen = input.length, inDim = weight.length, outDim = weight[0].length;
    const dWeight = this._zerosMatrix(inDim, outDim);
    const dInput = this._zerosMatrix(seqLen, inDim);
    for (let pos = 0; pos < seqLen; pos++) {
      const inRow = input[pos], dOutRow = dOutput[pos];
      for (let k = 0; k < inDim; k++) {
        let soma = 0;
        const wk = weight[k];
        for (let j = 0; j < outDim; j++) {
          dWeight[k][j] += inRow[k] * dOutRow[j];
          soma += dOutRow[j] * wk[j];
        }
        dInput[pos][k] = soma;
      }
    }
    return { dInput, dWeight };
  }

  // ==================== FORWARD (inferência, sem cache) ====================
  forward(tokenIndices) {
    const seqLen = tokenIndices.length;
    if (seqLen > this.blockSize) throw new Error('Sequência muito longa');

    let hidden = tokenIndices.map((token, pos) => {
      const tokEmb = this.tokenEmbedding[token];
      const posEmb = this.positionEmbedding[pos];
      return tokEmb.map((v, i) => v + posEmb[i]);
    });

    const headDim = this.dModel / this.numHeads;

    for (const layer of this.layers) {
      const Q = this._matmul(hidden, layer.wQ);
      const K = this._matmul(hidden, layer.wK);
      const V = this._matmul(hidden, layer.wV);
      const splitHeads = (matrix) => {
        const heads = [];
        for (let h = 0; h < this.numHeads; h++) heads.push(matrix.map(row => row.slice(h * headDim, (h + 1) * headDim)));
        return heads;
      };
      const Qh = splitHeads(Q), Kh = splitHeads(K), Vh = splitHeads(V);

      const attentionOutputs = [];
      for (let h = 0; h < this.numHeads; h++) {
        const scores = this._matmul(Qh[h], this._transpose(Kh[h]));
        const scaledScores = scores.map(row => row.map(v => v / Math.sqrt(headDim)));
        const maskedScores = scaledScores.map((row, i) => row.map((v, j) => (j > i ? -Infinity : v)));
        const attnWeights = maskedScores.map(row => this._softmax(row));
        attentionOutputs.push(this._matmul(attnWeights, Vh[h]));
      }
      let attnConcat = attentionOutputs[0].map((row, i) => {
        const concatRow = [];
        for (let h = 0; h < this.numHeads; h++) concatRow.push(...attentionOutputs[h][i]);
        return concatRow;
      });
      let attnOutput = this._matmul(attnConcat, layer.wO);
      const summed1 = hidden.map((row, i) => row.map((v, j) => v + attnOutput[i][j]));
      hidden = summed1.map(row => this._layerNormForward(row, layer.gamma1, layer.beta1).y);

      let ff = hidden.map(row => {
        const hiddenFF = this._matmul([row], layer.w1)[0].map((v, j) => this._relu(v + layer.b1[j]));
        return this._matmul([hiddenFF], layer.w2)[0].map((v, j) => v + layer.b2[j]);
      });
      const summed2 = hidden.map((row, i) => row.map((v, j) => v + ff[i][j]));
      hidden = summed2.map(row => this._layerNormForward(row, layer.gamma2, layer.beta2).y);
    }

    const lastHidden = hidden[hidden.length - 1];
    const logits = this._matmul([lastHidden], this.outputWeight)[0].map((v, i) => v + this.outputBias[i]);
    return this._softmax(logits);
  }

  // ==================== FORWARD DE TREINO (com cache p/ backward) ====================
  _forwardTreino(tokenIndices) {
    const seqLen = tokenIndices.length;
    if (seqLen > this.blockSize) throw new Error('Sequência muito longa');
    const headDim = this.dModel / this.numHeads;

    let hidden = tokenIndices.map((token, pos) => {
      const tokEmb = this.tokenEmbedding[token];
      const posEmb = this.positionEmbedding[pos];
      return tokEmb.map((v, i) => v + posEmb[i]);
    });

    const caches = [];
    for (const layer of this.layers) {
      const hiddenIn = hidden.map(r => [...r]);
      const Q = this._matmul(hidden, layer.wQ);
      const K = this._matmul(hidden, layer.wK);
      const V = this._matmul(hidden, layer.wV);
      const splitHeads = (m) => {
        const heads = [];
        for (let h = 0; h < this.numHeads; h++) heads.push(m.map(row => row.slice(h * headDim, (h + 1) * headDim)));
        return heads;
      };
      const Qh = splitHeads(Q), Kh = splitHeads(K), Vh = splitHeads(V);

      const attnWeights = [], headOutputs = [];
      for (let h = 0; h < this.numHeads; h++) {
        const scores = this._matmul(Qh[h], this._transpose(Kh[h]));
        const scaled = scores.map(row => row.map(v => v / Math.sqrt(headDim)));
        const masked = scaled.map((row, i) => row.map((v, j) => (j > i ? -Infinity : v)));
        const weights = masked.map(row => this._softmax(row));
        attnWeights.push(weights);
        headOutputs.push(this._matmul(weights, Vh[h]));
      }
      const attnConcat = headOutputs[0].map((row, i) => {
        const c = [];
        for (let h = 0; h < this.numHeads; h++) c.push(...headOutputs[h][i]);
        return c;
      });
      const attnOutput = this._matmul(attnConcat, layer.wO);
      const summed1 = hiddenIn.map((row, i) => row.map((v, j) => v + attnOutput[i][j]));
      const ln1 = summed1.map(row => this._layerNormForward(row, layer.gamma1, layer.beta1));
      const normed1 = ln1.map(r => r.y);

      const ffPreRelu = normed1.map(row => {
        const h = this._matmul([row], layer.w1)[0];
        return h.map((v, j) => v + layer.b1[j]);
      });
      const ffPostRelu = ffPreRelu.map(row => row.map(v => this._relu(v)));
      const ffOut = ffPostRelu.map(row => {
        const o = this._matmul([row], layer.w2)[0];
        return o.map((v, j) => v + layer.b2[j]);
      });
      const summed2 = normed1.map((row, i) => row.map((v, j) => v + ffOut[i][j]));
      const ln2 = summed2.map(row => this._layerNormForward(row, layer.gamma2, layer.beta2));
      const normed2 = ln2.map(r => r.y);

      caches.push({
        hiddenIn, Qh, Kh, Vh, attnWeights, headOutputs, attnConcat,
        summed1, mean1: ln1.map(r => r.mean), var1: ln1.map(r => r.variance), normed1,
        ffPreRelu, ffPostRelu,
        summed2, mean2: ln2.map(r => r.mean), var2: ln2.map(r => r.variance),
      });

      hidden = normed2;
    }

    const lastHidden = hidden[seqLen - 1];
    const logits = this._matmul([lastHidden], this.outputWeight)[0].map((v, i) => v + this.outputBias[i]);
    const probs = this._softmax(logits);
    return { probs, lastHidden, caches };
  }

  // ==================== TREINAMENTO (backprop real) ====================
  trainStep(tokenIndices, targetToken, learningRate = 0.01) {
    const { probs, lastHidden, caches } = this._forwardTreino(tokenIndices);
    const seqLen = tokenIndices.length;
    const headDim = this.dModel / this.numHeads;
    const clip = (v) => this._clip(v);

    // ---- gradiente da camada de saída ----
    const dLogits = probs.slice();
    dLogits[targetToken] -= 1; // derivada softmax + cross-entropy
    const { dInput: dLastHiddenArr, dWeight: dOutputWeight } = this._linearBackward([lastHidden], this.outputWeight, [dLogits]);
    const dOutputBias = dLogits;

    let dHidden = this._zerosMatrix(seqLen, this.dModel);
    dHidden[seqLen - 1] = dLastHiddenArr[0];

    const layerUpdates = [];

    for (let l = this.layers.length - 1; l >= 0; l--) {
      const cache = caches[l];
      const layer = this.layers[l];

      // LayerNorm 2 backward
      const dSummed2 = this._zerosMatrix(seqLen, this.dModel);
      const dGamma2 = this._zeros(this.dModel), dBeta2 = this._zeros(this.dModel);
      for (let pos = 0; pos < seqLen; pos++) {
        const { dx, dgamma, dbeta } = this._layerNormBackward(cache.summed2[pos], layer.gamma2, cache.mean2[pos], cache.var2[pos], dHidden[pos]);
        dSummed2[pos] = dx;
        for (let j = 0; j < this.dModel; j++) { dGamma2[j] += dgamma[j]; dBeta2[j] += dbeta[j]; }
      }
      const dFfOut = dSummed2; // residual: summed2 = normed1 + ffOut

      // Feed-forward backward
      const { dInput: dReluOut, dWeight: dW2 } = this._linearBackward(cache.ffPostRelu, layer.w2, dFfOut);
      const dB2 = this._sumRows(dFfOut);
      const dPreRelu = dReluOut.map((row, i) => row.map((v, j) => (cache.ffPreRelu[i][j] > 0 ? v : 0)));
      const { dInput: dNormed1FromFFN, dWeight: dW1 } = this._linearBackward(cache.normed1, layer.w1, dPreRelu);
      const dB1 = this._sumRows(dPreRelu);

      const dNormed1Total = dSummed2.map((row, i) => row.map((v, j) => v + dNormed1FromFFN[i][j]));

      // LayerNorm 1 backward
      const dSummed1 = this._zerosMatrix(seqLen, this.dModel);
      const dGamma1 = this._zeros(this.dModel), dBeta1 = this._zeros(this.dModel);
      for (let pos = 0; pos < seqLen; pos++) {
        const { dx, dgamma, dbeta } = this._layerNormBackward(cache.summed1[pos], layer.gamma1, cache.mean1[pos], cache.var1[pos], dNormed1Total[pos]);
        dSummed1[pos] = dx;
        for (let j = 0; j < this.dModel; j++) { dGamma1[j] += dgamma[j]; dBeta1[j] += dbeta[j]; }
      }
      const dAttnOutput = dSummed1; // residual: summed1 = hiddenIn + attnOutput

      // Projeção de saída da atenção (wO)
      const { dInput: dAttnConcat, dWeight: dWO } = this._linearBackward(cache.attnConcat, layer.wO, dAttnOutput);

      const dQ = this._zerosMatrix(seqLen, this.dModel);
      const dK = this._zerosMatrix(seqLen, this.dModel);
      const dV = this._zerosMatrix(seqLen, this.dModel);

      for (let h = 0; h < this.numHeads; h++) {
        const Vh = cache.Vh[h], Qh = cache.Qh[h], Kh = cache.Kh[h], attnW = cache.attnWeights[h];
        const dHeadOut = cache.headOutputs[h].map((row, pos) => dAttnConcat[pos].slice(h * headDim, (h + 1) * headDim));

        const dAttnW = this._zerosMatrix(seqLen, seqLen);
        const dVh = this._zerosMatrix(seqLen, headDim);
        for (let pos = 0; pos < seqLen; pos++) {
          for (let j = 0; j < seqLen; j++) {
            let dot = 0;
            for (let d = 0; d < headDim; d++) dot += dHeadOut[pos][d] * Vh[j][d];
            dAttnW[pos][j] = dot;
            const w = attnW[pos][j];
            if (w !== 0) for (let d = 0; d < headDim; d++) dVh[j][d] += w * dHeadOut[pos][d];
          }
        }
        // Softmax backward (jacobiano-vetor)
        const dScores = this._zerosMatrix(seqLen, seqLen);
        for (let pos = 0; pos < seqLen; pos++) {
          let sumTerm = 0;
          for (let j = 0; j < seqLen; j++) sumTerm += attnW[pos][j] * dAttnW[pos][j];
          for (let j = 0; j < seqLen; j++) dScores[pos][j] = attnW[pos][j] * (dAttnW[pos][j] - sumTerm);
        }
        const dQh = this._zerosMatrix(seqLen, headDim);
        const dKh = this._zerosMatrix(seqLen, headDim);
        const invSqrt = 1 / Math.sqrt(headDim);
        for (let pos = 0; pos < seqLen; pos++) {
          for (let j = 0; j < seqLen; j++) {
            const factor = dScores[pos][j] * invSqrt;
            if (factor === 0) continue;
            for (let d = 0; d < headDim; d++) {
              dQh[pos][d] += factor * Kh[j][d];
              dKh[j][d] += factor * Qh[pos][d];
            }
          }
        }
        for (let pos = 0; pos < seqLen; pos++) {
          for (let d = 0; d < headDim; d++) {
            dQ[pos][h * headDim + d] = dQh[pos][d];
            dK[pos][h * headDim + d] = dKh[pos][d];
            dV[pos][h * headDim + d] = dVh[pos][d];
          }
        }
      }

      const { dInput: dHiddenInQ, dWeight: dWQ } = this._linearBackward(cache.hiddenIn, layer.wQ, dQ);
      const { dInput: dHiddenInK, dWeight: dWK } = this._linearBackward(cache.hiddenIn, layer.wK, dK);
      const { dInput: dHiddenInV, dWeight: dWV } = this._linearBackward(cache.hiddenIn, layer.wV, dV);

      const dHiddenIn = this._zerosMatrix(seqLen, this.dModel);
      for (let pos = 0; pos < seqLen; pos++)
        for (let k = 0; k < this.dModel; k++)
          dHiddenIn[pos][k] = dSummed1[pos][k] + dHiddenInQ[pos][k] + dHiddenInK[pos][k] + dHiddenInV[pos][k];

      layerUpdates.push({ dWQ, dWK, dWV, dWO, dW1, dB1, dW2, dB2, dGamma1, dBeta1, dGamma2, dBeta2 });
      dHidden = dHiddenIn;
    }
    layerUpdates.reverse();

    // ---- aplicar atualização (SGD com clipping) ----
    const applyMat = (param, grad) => { for (let i = 0; i < param.length; i++) for (let j = 0; j < param[i].length; j++) param[i][j] -= learningRate * clip(grad[i][j]); };
    const applyVec = (param, grad) => { for (let i = 0; i < param.length; i++) param[i] -= learningRate * clip(grad[i]); };

    applyMat(this.outputWeight, dOutputWeight);
    applyVec(this.outputBias, dOutputBias);

    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const g = layerUpdates[l];
      applyMat(layer.wQ, g.dWQ); applyMat(layer.wK, g.dWK); applyMat(layer.wV, g.dWV); applyMat(layer.wO, g.dWO);
      applyMat(layer.w1, g.dW1); applyVec(layer.b1, g.dB1);
      applyMat(layer.w2, g.dW2); applyVec(layer.b2, g.dB2);
      applyVec(layer.gamma1, g.dGamma1); applyVec(layer.beta1, g.dBeta1);
      applyVec(layer.gamma2, g.dGamma2); applyVec(layer.beta2, g.dBeta2);
    }

    for (let pos = 0; pos < seqLen; pos++) {
      const token = tokenIndices[pos];
      for (let j = 0; j < this.dModel; j++) {
        this.tokenEmbedding[token][j] -= learningRate * clip(dHidden[pos][j]);
        this.positionEmbedding[pos][j] -= learningRate * clip(dHidden[pos][j]);
      }
    }

    return -Math.log(probs[targetToken] + 1e-10); // cross-entropy loss deste passo
  }

  // ==================== GERAÇÃO ====================
  generate(promptIndices, maxTokens = 100, temperature = 0.8) {
    const generated = [...promptIndices];
    for (let i = 0; i < maxTokens; i++) {
      const input = generated.slice(-this.blockSize);
      const probs = this.forward(input);
      const logits = probs.map(p => Math.log(p + 1e-10) / temperature);
      const temperedProbs = this._softmax(logits);
      const r = Math.random();
      let cum = 0, nextToken = 0;
      for (let j = 0; j < this.vocabSize; j++) {
        cum += temperedProbs[j];
        if (r <= cum) { nextToken = j; break; }
      }
      generated.push(nextToken);
    }
    return generated;
  }

  // ==================== PERSISTÊNCIA ====================
  salvar() {
    return JSON.stringify({
      vocabSize: this.vocabSize,
      dModel: this.dModel,
      numHeads: this.numHeads,
      numLayers: this.numLayers,
      blockSize: this.blockSize,
      vocabChars: this.vocabChars,
      tokenEmbedding: this.tokenEmbedding,
      positionEmbedding: this.positionEmbedding,
      layers: this.layers,
      outputWeight: this.outputWeight,
      outputBias: this.outputBias,
    });
  }

  carregar(json) {
    const data = JSON.parse(json);
    Object.assign(this, data);
  }
}

if (typeof window !== 'undefined') {
  window.MiniTransformer = MiniTransformer;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = MiniTransformer;
}
