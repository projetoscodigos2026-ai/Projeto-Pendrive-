Minha Própria IA
Um projeto de IA feito do zero, em JavaScript puro, rodando 100% offline no navegador do celular — sem servidor, sem conta, sem internet depois de carregado.
Ela tem três partes que trabalham juntas:
Arquivo
O que é
MinhaBibliotecaIA.js
Rede neural feedforward (números → números). Boa pra classificação, XOR, lógica.
CerebroConversa.js
Roteador de intenções por regras — reconhece saudações, perguntas sobre si mesma, etc.
MiniTransformer.js
Modelo de linguagem em nível de caractere, com backpropagation completa de verdade (atenção, feed-forward, layer norm e embeddings todos aprendem).
index.html
O corpo — interface única que junta os três cérebros.
Quando você conversa com ela, a ordem de decisão é:
Números? → vai pra rede neural numérica.
Frase conhecida (oi, quem é você, ajuda...) → resposta certeira por regra.
Texto livre, modelo de linguagem treinado → ela tenta continuar o texto, letra por letra (modo experimental, avisado na interface).
Nada disso → ela admite que não sabe e te diz o que fazer.
O que ela é (de verdade) e o que não é
É uma IA real: os pesos mudam com o treino, você vê a perda caindo no log, e o comportamento muda conforme ela aprende.
Não é um ChatGPT: o modelo de linguagem é pequeníssimo (poucos milhares de parâmetros) e nível de caractere. Ele aprende padrões locais do texto que você treinar — não vai escrever parágrafos com sentido, mas vai imitar o estilo, palavras e ortografia do que você ensinar.
Passo a passo pra subir no GitHub (só com o celular)
Crie a conta e o repositório
Abra github.com no navegador do celular (ou o app oficial).
Toque em "New repository", dê um nome (ex: minha-ia), marque como público ou privado, crie.
Suba os 4 arquivos
Na página do repositório, toque em "Add file" → "Upload files".
Selecione os 4 arquivos: index.html, MinhaBibliotecaIA.js, CerebroConversa.js, MiniTransformer.js.
Escreva uma mensagem de commit (ex: "primeira versão da minha IA") e confirme.
Ative o GitHub Pages (pra ter um link público onde ela roda de verdade)
No repositório: Settings → Pages.
Em "Branch", escolha main e pasta / (root), salve.
Espere 1-2 minutos. O link vai aparecer no topo da mesma página (algo como https://seu-usuario.github.io/minha-ia/).
Esse link já é sua IA funcionando, acessível de qualquer navegador.
Para editar depois, sem precisar de computador
Troque github.com por github.dev na URL do repositório (ex: github.dev/seu-usuario/minha-ia).
Abre um VS Code completo no navegador do celular. Edite os arquivos, aperte Ctrl+S (ou o botão de salvar), depois vá no ícone de "Source Control" (os três pontinhos ramificados) pra fazer commit direto.
Backup extra no pendrive
Baixe os arquivos do GitHub (botão "Code" → "Download ZIP") de vez em quando.
Copie pro pendrive via app de arquivos do Android (se seu aparelho suportar OTG — um adaptador USB-C/micro-USB pra pendrive).
Testando rápido depois de publicado
Rede numérica (aba "Treinar Números") — cole isto e treine (XOR, bom teste de aprendizado):

Código
0,0,0
0,1,1
1,0,1
1,1,0

Arquitetura: entrada 2, ocultos 4, saída 1, sigmoide, ~2000 épocas.
Modelo de linguagem (aba "Treinar Linguagem") — carregue qualquer .txt curto (até ~5-10 mil caracteres pra treinar rápido no celular), inicialize com Dim 32 / Heads 2 / Layers 2 / Block 16, treine por 5-10 épocas e observe a perda caindo no log verde. Se ela cair, funcionou.
Próximos passos possíveis (quando quiser voltar)
Mais intenções em CerebroConversa.js — é só editar o array intencoes.
Treinar o modelo de linguagem com textos maiores (mais épocas, mais paciência).
Conectar a interface a uma API de LLM de verdade pra conversa livre, mantendo os seus modelos próprios para tarefas específicas — um "cérebro híbrido" mais avançado.
