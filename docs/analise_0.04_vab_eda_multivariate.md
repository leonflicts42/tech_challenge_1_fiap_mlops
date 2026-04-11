O que isso faz (nível profissional)
📊 1. Heatmap
Visual geral de dependência
Identifica padrões visuais
🎯 2. Correlação com target

Você descobre:

Variáveis mais relevantes
Direção da relação
⚠️ 3. Multicolinearidade (corr > 0.7)

Problema:

Modelos lineares instáveis
Interpretação ruim
🔥 4. VIF (nível banco / indústria)
VIF	Interpretação
1	ok
5+	atenção
10+	problema sério

🧠 1. COMO INTERPRETAR SEU LOG (correlação + VIF)
📊 Correlação com TARGET
tenure | corr_target=-0.35
monthly_charges | corr_target=0.19
👉 Interpretação:
|corr| < 0.1 → irrelevante
0.1 – 0.3 → fraca (mas útil)
0.3 – 0.5 → moderada
> 0.5 → forte

✔ No churn:

tenure geralmente é muito relevante (negativo)
monthly_charges → impacto positivo moderado
⚠️ Multicolinearidade (corr entre features)
tenure vs total_charges | corr=0.85
👉 Interpretação:

0.7 → risco de redundância

0.9 → praticamente duplicadas

✔ No seu dataset:

total_charges ≈ tenure × monthly_charges

👉 Isso NÃO é erro — é relação estrutural

🔥 VIF (o mais importante)
total_charges | VIF=12.4 (ALTO)
tenure | VIF=6.2 (MODERADO)
monthly_charges | VIF=3.1 (OK)
👉 Interpretação:
VIF	Ação
< 5	ok
5–10	atenção
> 10	remover ou tratar
🎯 Decisão prática

👉 Regra de ouro:

Se duas variáveis são altamente correlacionadas:

mantenha a mais interpretável
ou a mais preditiva

✔ Exemplo real:

manter tenure
remover total_charges

🎯 3. O que você DEVE fazer (nível indústria)
✔ Feature selection simples e correta
Remover: TotalCharges
Manter: tenure, monthly_charges

✔ Focar no que realmente importa
📊 Variáveis categóricas (muito mais importantes)

No dataset Telco:

Contract 🔥
Internet Service 🔥
Payment Method 🔥

👉 essas têm MUITO mais poder que numéricas

🧠 Insight nível senior

👉 Em churn de telco:

numéricas → ajudam
categóricas → dominam o modelo