// ==========================================
// CerebroConversa.js
// Camada de conversa por regras + intenção.
// NÃO é um modelo de linguagem — é um roteador
// de intenções simples que dá "personalidade"
// à interface, e delega números pra rede neural.
// ==========================================

class CerebroConversa {
  constructor(identidade) {
    this.identidade = identidade;
    this.intencoes = [
      {
        nome: 'saudacao',
        padroes: [/^oi\b/i, /^ol[aá]\b/i, /^e a[íi]\b/i, /bom dia/i, /boa tarde/i, /boa noite/i],
        respostas: () => [`Oi! Eu sou a ${this.identidade.nome} ${this.identidade.avatar}. Já treinei por ${this.identidade.epocasTreinadas} épocas até agora.`]
      },
      {
        nome: 'quem_e_voce',
        padroes: [/quem (é|e) voc[eê]/i, /seu nome/i, /o que voc[eê] (é|e)/i],
        respostas: () => [`Eu sou a ${this.identidade.nome}. Fui criada em ${new Date(this.identidade.criadaEm).toLocaleDateString('pt-BR')} como uma rede neural feedforward — ainda pequena, mas crescendo a cada treino.`]
      },
      {
        nome: 'ajuda',
        padroes: [/ajuda/i, /como (funciona|uso)/i, /o que (voc[eê] )?faz/i],
        respostas: () => ['Se você digitar números separados por vírgula (ex: "1,0"), eu uso minha rede neural pra prever uma saída. Se digitar uma pergunta, eu tento reconhecer a intenção e responder.']
      },
      {
        nome: 'despedida',
        padroes: [/^tchau/i, /at[eé] mais/i, /falou/i],
        respostas: () => ['Até mais! Vou guardar isso na memória.']
      },
      {
        nome: 'elogio',
        padroes: [/voc[eê] (é|e) (legal|incr[íi]vel|top|show)/i, /gostei de voc[eê]/i],
        respostas: () => ['Obrigada! Cada treino me ajuda a ficar um pouco melhor. 🧠✨']
      }
    ];
  }

  // Tenta reconhecer uma intenção por regras. Retorna string de resposta ou null se não reconhecer.
  responder(texto) {
    for (const intencao of this.intencoes) {
      if (intencao.padroes.some(p => p.test(texto))) {
        const opcoes = intencao.respostas();
        return opcoes[Math.floor(Math.random() * opcoes.length)];
      }
    }
    return null; // sinaliza "não reconheci, tente a rede neural ou mostre fallback"
  }

  // Detecta se o texto parece ser uma lista de números pra rede neural
  ehEntradaNumerica(texto) {
    const partes = texto.split(',').map(p => p.trim());
    return partes.length > 0 && partes.every(p => p !== '' && !isNaN(parseFloat(p)));
  }
}

if (typeof window !== 'undefined') {
  window.CerebroConversa = CerebroConversa;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = CerebroConversa;
}
