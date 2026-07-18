// ==========================================
// MinhaBibliotecaIA.js  (v2)
// Rede neural feedforward com backpropagation
// Agora com: ativações escolhíveis, momentum,
// identidade e memória (a "alma" da IA)
// Feita para rodar 100% offline no navegador
// ==========================================

class MinhaBibliotecaIA {
  constructor(camadas, opcoes = {}) {
    // camadas: array com número de neurônios em cada camada (ex: [2, 4, 1])
    this.camadas = camadas;
    this.pesos = [];
    this.vieses = [];
    this.velPesos = [];   // velocidade (momentum) dos pesos
    this.velVieses = [];  // velocidade (momentum) dos vieses
    this.taxaAprendizado = opcoes.taxaAprendizado ?? 0.1;
    this.momentum = opcoes.momentum ?? 0.9;
    this.ativacao = opcoes.ativacao ?? 'sigmoide'; // 'sigmoide' | 'tanh' | 'relu'

    // Identidade e memória: a parte "viva" da IA
    this.identidade = {
      nome: opcoes.nome ?? 'Minha IA',
      avatar: opcoes.avatar ?? '🧠',
      criadaEm: opcoes.criadaEm ?? new Date().toISOString(),
      epocasTreinadas: 0
    };
    this.memoria = []; // histórico de interações: {entrada, saida, quando}

    for (let i = 0; i < camadas.length - 1; i++) {
      this.pesos.push(this._matrizAleatoria(camadas[i + 1], camadas[i]));
      this.vieses.push(this._matrizAleatoria(camadas[i + 1], 1));
      this.velPesos.push(this._matrizZeros(camadas[i + 1], camadas[i]));
      this.velVieses.push(this._matrizZeros(camadas[i + 1], 1));
    }
  }

  // ---------- Funções de ativação ----------
  _ativar(x) {
    switch (this.ativacao) {
      case 'tanh': return Math.tanh(x);
      case 'relu': return Math.max(0, x);
      default: return 1.0 / (1.0 + Math.exp(-x)); // sigmoide
    }
  }

  // derivada em função do valor JÁ ativado (y = ativar(x))
  _dativar(y) {
    switch (this.ativacao) {
      case 'tanh': return 1 - y * y;
      case 'relu': return y > 0 ? 1 : 0;
      default: return y * (1.0 - y); // sigmoide
    }
  }

  // ---------- Álgebra de matrizes ----------
  _matrizAleatoria(linhas, colunas) {
    // inicialização estilo Xavier: reduz risco de saturação
    const limite = Math.sqrt(6 / (linhas + colunas));
    return Array.from({ length: linhas }, () =>
      Array.from({ length: colunas }, () => (Math.random() * 2 - 1) * limite)
    );
  }

  _matrizZeros(linhas, colunas) {
    return Array.from({ length: linhas }, () => Array(colunas).fill(0));
  }

  _produtoMatricial(a, b) {
    const linhasA = a.length, colunasA = a[0].length, colunasB = b[0].length;
    const resultado = Array.from({ length: linhasA }, () => Array(colunasB).fill(0));
    for (let i = 0; i < linhasA; i++) {
      for (let j = 0; j < colunasB; j++) {
        for (let k = 0; k < colunasA; k++) {
          resultado[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    return resultado;
  }

  _adicionarVies(matriz, vies) {
    return matriz.map((linha, i) => linha.map((val) => val + vies[i][0]));
  }

  _transpor(matriz) {
    const linhas = matriz.length, colunas = matriz[0].length;
    return Array.from({ length: colunas }, (_, i) =>
      Array.from({ length: linhas }, (_, j) => matriz[j][i])
    );
  }

  // ---------- Previsão ----------
  prever(entrada) {
    let ativacao = entrada.map(v => [v]);
    for (let i = 0; i < this.pesos.length; i++) {
      const z = this._adicionarVies(this._produtoMatricial(this.pesos[i], ativacao), this.vieses[i]);
      ativacao = z.map(linha => linha.map(v => this._ativar(v)));
    }
    return ativacao.map(linha => linha[0]);
  }

  // ---------- Treino (um exemplo) com momentum ----------
  treinar(entrada, alvo) {
    let ativacao = entrada.map(v => [v]);
    const alvoMatriz = alvo.map(v => [v]);
    const ativacoes = [ativacao];

    for (let i = 0; i < this.pesos.length; i++) {
      const z = this._adicionarVies(this._produtoMatricial(this.pesos[i], ativacao), this.vieses[i]);
      ativacao = z.map(linha => linha.map(v => this._ativar(v)));
      ativacoes.push(ativacao);
    }

    let delta = ativacoes[ativacoes.length - 1].map((linha, i) =>
      linha.map((val, j) => (val - alvoMatriz[i][j]) * this._dativar(val))
    );

    for (let l = this.pesos.length - 1; l >= 0; l--) {
      const ativacaoAnterior = ativacoes[l];
      const gradPesos = this._produtoMatricial(delta, this._transpor(ativacaoAnterior));
      const gradVieses = delta;

      for (let i = 0; i < this.pesos[l].length; i++) {
        for (let j = 0; j < this.pesos[l][i].length; j++) {
          // momentum: v = m*v - taxa*grad ; peso += v
          this.velPesos[l][i][j] = this.momentum * this.velPesos[l][i][j] - this.taxaAprendizado * gradPesos[i][j];
          this.pesos[l][i][j] += this.velPesos[l][i][j];
        }
        this.velVieses[l][i][0] = this.momentum * this.velVieses[l][i][0] - this.taxaAprendizado * gradVieses[i][0];
        this.vieses[l][i][0] += this.velVieses[l][i][0];
      }

      if (l > 0) {
        const pesosTranspostos = this._transpor(this.pesos[l]);
        delta = this._produtoMatricial(pesosTranspostos, delta).map((linha, i) =>
          linha.map((val, j) => val * this._dativar(ativacoes[l][i][j]))
        );
      }
    }
  }

  // Treina um lote (array de {entrada, alvo}) e conta como parte das "épocas vividas"
  treinarLote(dados) {
    for (const d of dados) this.treinar(d.entrada, d.alvo);
    this.identidade.epocasTreinadas++;
  }

  // ---------- Memória (a "alma") ----------
  lembrar(entrada, saida) {
    this.memoria.push({ entrada, saida, quando: new Date().toISOString() });
    if (this.memoria.length > 200) this.memoria.shift(); // limite pra não crescer infinito
  }

  // ---------- Persistência ----------
  salvar() {
    return JSON.stringify({
      camadas: this.camadas,
      pesos: this.pesos,
      vieses: this.vieses,
      velPesos: this.velPesos,
      velVieses: this.velVieses,
      ativacao: this.ativacao,
      taxaAprendizado: this.taxaAprendizado,
      momentum: this.momentum,
      identidade: this.identidade,
      memoria: this.memoria
    });
  }

  carregar(json) {
    const d = JSON.parse(json);
    this.camadas = d.camadas;
    this.pesos = d.pesos;
    this.vieses = d.vieses;
    this.velPesos = d.velPesos ?? this.pesos.map(p => this._matrizZeros(p.length, p[0].length));
    this.velVieses = d.velVieses ?? this.vieses.map(v => this._matrizZeros(v.length, 1));
    this.ativacao = d.ativacao ?? 'sigmoide';
    this.taxaAprendizado = d.taxaAprendizado ?? 0.1;
    this.momentum = d.momentum ?? 0.9;
    this.identidade = d.identidade ?? { nome: 'Minha IA', avatar: '🧠', criadaEm: new Date().toISOString(), epocasTreinadas: 0 };
    this.memoria = d.memoria ?? [];
  }
}

if (typeof window !== 'undefined') {
  window.MinhaBibliotecaIA = MinhaBibliotecaIA;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = MinhaBibliotecaIA;
}
