from pathlib import Path

import importlib
import subprocess
import sys


def ensure_dependencies() -> None:
	required_packages = {
		"numpy": "numpy",
		"pandas": "pandas",
		"plotly": "plotly",
	}

	for module_name, package_name in required_packages.items():
		try:
			importlib.import_module(module_name)
		except ImportError:
			subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


ensure_dependencies()

import numpy as np
import pandas as pd
from plotly import graph_objects as go


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	csv_path = base_dir / "../../../dataset/data.csv"

	df = pd.read_csv(csv_path)

	# Colunas de gasto direto (despesas mensais)
	expense_cols = [
		"Rent",
		"Loan_Repayment",
		"Insurance",
		"Groceries",
		"Transport",
		"Eating_Out",
		"Entertainment",
		"Utilities",
		"Healthcare",
		"Education",
		"Miscellaneous",
	]

	df["Gasto_Total"] = df[expense_cols].sum(axis=1)
	df["Taxa_Poupanca_Real"] = (df["Disposable_Income"] / df["Income"]) * 100

	age_profile = (
		df.groupby("Age", as_index=False)
		.agg(
			gasto_medio=("Gasto_Total", "mean"),
			poupanca_media=("Taxa_Poupanca_Real", "mean"),
			renda_media=("Income", "mean"),
			dependentes_medios=("Dependents", "mean"),
			emprestimo_medio=("Loan_Repayment", "mean"),
			educacao_media=("Education", "mean"),
			n=("Age", "size"),
		)
		.sort_values("Age")
	)

	age_profile["gasto_medio_suave"] = (
		age_profile["gasto_medio"].rolling(window=5, center=True, min_periods=1).mean()
	)

	# Ajuste quadratico para estimar o ponto onde o gasto deixa de subir
	x = age_profile["Age"].to_numpy()
	y = age_profile["gasto_medio_suave"].to_numpy()

	coef = np.polyfit(x, y, deg=2)
	a, b, c = coef

	if abs(a) > 1e-12:
		idade_virada = -b / (2 * a)
		idade_virada = float(np.clip(idade_virada, x.min(), x.max()))
	else:
		idade_virada = float(np.median(x))

	idx_maturidade = int(np.argmin(np.abs(x - idade_virada)))
	idade_maturidade = int(x[idx_maturidade])
	gasto_maturidade = float(y[idx_maturidade])

	fig = go.Figure()
	fig.add_trace(
		go.Scatter(
			x=age_profile["Age"],
			y=age_profile["gasto_medio"],
			mode="lines+markers",
			name="Gasto medio por idade (bruto)",
			opacity=0.35,
		)
	)
	fig.add_trace(
		go.Scatter(
			x=age_profile["Age"],
			y=age_profile["gasto_medio_suave"],
			mode="lines",
			name="Tendencia suavizada",
			line={"width": 3},
		)
	)
	fig.add_trace(
		go.Scatter(
			x=[idade_maturidade],
			y=[gasto_maturidade],
			mode="markers",
			name="Ponto de maturidade financeira",
			marker={"color": "red", "size": 12},
		)
	)

	fig.add_vline(x=idade_maturidade, line_dash="dash", line_color="gray")
	fig.update_layout(
		title="Evolucao do gasto medio com a idade",
		xaxis_title="Idade",
		yaxis_title="Gasto total medio",
		template="plotly_white",
	)
	fig.show()

	print(f"Idade estimada de maturidade financeira: {idade_maturidade} anos")
	print(f"Gasto medio nesse ponto: {gasto_maturidade:,.2f}")
	print(f"Coeficientes da curva quadratica (a, b, c): {coef}")

	antes = df[df["Age"] < idade_maturidade]
	depois = df[df["Age"] >= idade_maturidade]

	def resumo_segmento(base: pd.DataFrame) -> pd.Series:
		return pd.Series(
			{
				"gasto_total_medio": base["Gasto_Total"].mean(),
				"taxa_poupanca_real_media_%": base["Taxa_Poupanca_Real"].mean(),
				"dependentes_medios": base["Dependents"].mean(),
				"emprestimo_medio": base["Loan_Repayment"].mean(),
				"educacao_media": base["Education"].mean(),
				"saude_media": base["Healthcare"].mean(),
				"renda_media": base["Income"].mean(),
			}
		)

	comparativo = pd.DataFrame(
		{
			"Antes da maturidade": resumo_segmento(antes),
			"Depois da maturidade": resumo_segmento(depois),
		}
	)

	faixas = pd.cut(df["Age"], bins=[18, 30, 40, 50, 60, 70], right=False)
	faixa_gasto = (
		df.assign(Faixa_Idade=faixas)
		.groupby("Faixa_Idade", observed=False)["Gasto_Total"]
		.mean()
		.to_frame("gasto_total_medio")
	)

	print("\nComparativo antes/depois da maturidade:")
	print(comparativo.round(2).to_string())

	print("\nGasto medio por faixa etaria:")
	print(faixa_gasto.round(2).to_string())

	tendencia = (
		"aumenta ate um ponto e depois desacelera/diminui"
		if a < 0
		else "mantem tendencia de alta no intervalo observado"
	)

	print("\nConclusao:")
	print(f"- Tendencia do gasto com a idade: {tendencia}.")
	print(f"- Ponto estimado de maturidade financeira: {idade_maturidade} anos.")
	print("- Possiveis causas para mudanca apos a maturidade:")
	print("  1) reducao relativa de gastos com educacao e/ou emprestimos;")
	print("  2) maior controle de consumo discricionario;")
	print("  3) aumento da eficiencia financeira com maior experiencia e planejamento.")


if __name__ == "__main__":
	main()
