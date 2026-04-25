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
from plotly.subplots import make_subplots


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	csv_path = base_dir / "../../../dataset/data.csv"

	df = pd.read_csv(csv_path)

	# Proxy de "gasto com dopamina" / "Imposto do Cansaco":
	# consumo mais impulsivo ou de alivio imediato, normalmente associado a refeicoes fora,
	# entretenimento e gastos diversos.
	dopamine_cols = ["Eating_Out", "Entertainment", "Miscellaneous"]
	df["Gasto_Dopamina"] = df[dopamine_cols].sum(axis=1)
	df["Imposto_do_Cansaco_%"] = (df["Gasto_Dopamina"] / df["Income"]) * 100
	df["Gasto_Dopamina_por_Dependentes"] = df["Gasto_Dopamina"] / (df["Dependents"] + 1)

	occupation_profile = (
		df.groupby("Occupation", as_index=False)
		.agg(
			gasto_dopamina_medio=("Gasto_Dopamina", "mean"),
			imposto_cansaco_medio=("Imposto_do_Cansaco_%", "mean"),
			renda_media=("Income", "mean"),
			despesa_total_media=("Rent", "mean"),
			loan_medio=("Loan_Repayment", "mean"),
			dependentes_medios=("Dependents", "mean"),
			n=("Occupation", "size"),
		)
		.sort_values("gasto_dopamina_medio", ascending=False)
	)

	media_gasto_dopamina = df["Gasto_Dopamina"].mean()
	media_imposto_cansaco = df["Imposto_do_Cansaco_%"].mean()

	occupation_profile["participacao_na_renda_%"] = (
		occupation_profile["gasto_dopamina_medio"] / occupation_profile["renda_media"] * 100
	)
	occupation_profile["desvio_gasto_%"] = (
		(occupation_profile["gasto_dopamina_medio"] / media_gasto_dopamina - 1) * 100
	)
	occupation_profile["desvio_imposto_%"] = (
		(occupation_profile["imposto_cansaco_medio"] / media_imposto_cansaco - 1) * 100
	)

	top_occupation = occupation_profile.iloc[0]
	bottom_occupation = occupation_profile.iloc[-1]

	plot_profile = occupation_profile.sort_values("desvio_gasto_%")

	def descreve_desvio(valor: float) -> str:
		return "acima da media" if valor >= 0 else "abaixo da media"

	fig = make_subplots(
		rows=1,
		cols=2,
		subplot_titles=(
			"Desvio do gasto com dopamina vs media geral",
			"Desvio do imposto do cansaco vs media geral",
		),
	)

	fig.add_trace(
		go.Bar(
			x=plot_profile["desvio_gasto_%"],
			y=plot_profile["Occupation"],
			name="Desvio Gasto Dopamina",
			orientation="h",
			marker_color=["#D64541" if valor < 0 else "#2D6CDF" for valor in plot_profile["desvio_gasto_%"]],
			text=plot_profile["desvio_gasto_%"].round(2).astype(str) + "%",
			textposition="outside",
		),
		row=1,
		col=1,
	)

	fig.add_trace(
		go.Bar(
			x=plot_profile["desvio_imposto_%"],
			y=plot_profile["Occupation"],
			name="Desvio Imposto do Cansaco",
			orientation="h",
			marker_color=["#D64541" if valor < 0 else "#F28E2B" for valor in plot_profile["desvio_imposto_%"]],
			text=plot_profile["desvio_imposto_%"].round(2).astype(str) + "%",
			textposition="outside",
		),
		row=1,
		col=2,
	)

	fig.update_layout(
		title="Diferença relativa do gasto com dopamina por ocupação",
		template="plotly_white",
		height=600,
		showlegend=False,
	)
	fig.update_xaxes(title_text="Desvio percentual", zeroline=True, zerolinewidth=2, zerolinecolor="gray", row=1, col=1)
	fig.update_xaxes(title_text="Desvio percentual", zeroline=True, zerolinewidth=2, zerolinecolor="gray", row=1, col=2)
	fig.update_yaxes(title_text="Ocupacao", row=1, col=1)
	fig.update_yaxes(title_text="Ocupacao", row=1, col=2)
	fig.show()

	print("Resumo por ocupacao (ordenado pelo gasto dopamina medio):")
	print(
		occupation_profile[
			[
				"Occupation",
				"gasto_dopamina_medio",
				"imposto_cansaco_medio",
				"participacao_na_renda_%",
				"renda_media",
				"loan_medio",
				"dependentes_medios",
				"n",
			]
		].round(2).to_string(index=False)
	)

	print("\nLeitura executiva:")
	print(
		f"- A media geral de gasto com dopamina e {media_gasto_dopamina:,.2f}, ou {media_imposto_cansaco:.2f}% da renda."
	)
	print(
		f"- A ocupacao que mais gasta com dopamina e {top_occupation['Occupation']} ({top_occupation['gasto_dopamina_medio']:,.2f} em media, {abs(top_occupation['desvio_gasto_%']):.2f}% acima da media)."
	)
	print(
		f"- A ocupacao que menos gasta e {bottom_occupation['Occupation']} ({bottom_occupation['gasto_dopamina_medio']:,.2f} em media, {abs(bottom_occupation['desvio_gasto_%']):.2f}% {descreve_desvio(bottom_occupation['desvio_gasto_%'])})."
	)
	print("- O 'Imposto do Cansaco' tende a subir quando a renda nao cresce na mesma velocidade que o gasto discrecionario.")
	print("- Ocupacoes com maior pressao de tempo/estresse tendem a exibir maior gasto em refeicoes fora e entretenimento, mas isso deve ser interpretado como correlacao, nao causalidade direta.")

	print("\nTabela de desvios relativos (%):")
	print(
		occupation_profile[
			[
				"Occupation",
				"gasto_dopamina_medio",
				"desvio_gasto_%",
				"imposto_cansaco_medio",
				"desvio_imposto_%",
			]
		].round(2).to_string(index=False)
	)

	# Corte simples para destacar as ocupacoes mais expostas
	top5 = occupation_profile.head(5)[["Occupation", "gasto_dopamina_medio", "imposto_cansaco_medio"]].round(2)
	print("\nTop 5 ocupacoes por gasto dopamina medio:")
	print(top5.to_string(index=False))


if __name__ == "__main__":
	main()
