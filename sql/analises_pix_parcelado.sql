-- Análises em SQL sobre a base de Pix parcelado (tabela: pix_parcelado)
-- Colunas: cliente_id, idade, renda_estimada, valor_compra, qtd_parcelas,
--          ja_usou_pix_parcelado_antes, score_credito, inadimplente

-- 1) Taxa de inadimplência geral, só pra ter um número de partida
SELECT
    COUNT(*) AS total_compras,
    SUM(inadimplente) AS total_atrasou,
    ROUND(100.0 * SUM(inadimplente) / COUNT(*), 1) AS taxa_inadimplencia_pct
FROM pix_parcelado;


-- 2) Inadimplência por quantidade de parcelas
-- (queria ver se parcelar mais vezes aumenta o risco de atraso)
SELECT
    qtd_parcelas,
    COUNT(*) AS total_compras,
    ROUND(100.0 * SUM(inadimplente) / COUNT(*), 1) AS taxa_inadimplencia_pct
FROM pix_parcelado
GROUP BY qtd_parcelas
ORDER BY qtd_parcelas;


-- 3) Compara cliente novo x cliente que já usou o produto antes
SELECT
    CASE WHEN ja_usou_pix_parcelado_antes = 1 THEN 'Já usou antes' ELSE 'Primeira vez' END AS perfil_cliente,
    COUNT(*) AS total_compras,
    ROUND(100.0 * SUM(inadimplente) / COUNT(*), 1) AS taxa_inadimplencia_pct
FROM pix_parcelado
GROUP BY perfil_cliente;


-- 4) Inadimplência por faixa de score (pra pensar em política de limite/aprovação)
SELECT
    CASE
        WHEN score_credito < 500 THEN 'até 499'
        WHEN score_credito < 650 THEN '500 a 649'
        WHEN score_credito < 800 THEN '650 a 799'
        ELSE '800+'
    END AS faixa_score,
    COUNT(*) AS total_compras,
    ROUND(100.0 * SUM(inadimplente) / COUNT(*), 1) AS taxa_inadimplencia_pct,
    ROUND(AVG(valor_compra), 2) AS ticket_medio
FROM pix_parcelado
GROUP BY faixa_score
ORDER BY faixa_score;


-- 5) Quanto do "comprometimento de renda" (valor da compra / renda) tá relacionado
-- com o atraso -- aqui uso uma faixa simples pra não precisar calcular por linha
SELECT
    CASE
        WHEN valor_compra / renda_estimada < 0.05 THEN 'baixo (< 5% da renda)'
        WHEN valor_compra / renda_estimada < 0.15 THEN 'médio (5% a 15%)'
        ELSE 'alto (> 15% da renda)'
    END AS faixa_comprometimento,
    COUNT(*) AS total_compras,
    ROUND(100.0 * SUM(inadimplente) / COUNT(*), 1) AS taxa_inadimplencia_pct
FROM pix_parcelado
GROUP BY faixa_comprometimento;


-- 6) Teste A/B: qual lembrete de cobrança recupera mais cliente inadimplente?
-- (só considera quem realmente ficou inadimplente e recebeu algum lembrete)
SELECT
    estrategia_lembrete,
    COUNT(*) AS total_clientes,
    SUM(recuperou) AS total_recuperou,
    ROUND(100.0 * SUM(recuperou) / COUNT(*), 1) AS taxa_recuperacao_pct
FROM pix_parcelado
WHERE inadimplente = 1
  AND estrategia_lembrete <> 'N/A'
GROUP BY estrategia_lembrete
ORDER BY taxa_recuperacao_pct DESC;
