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
	csv_path = base_dir / "../data.csv"

	df = pd.read_csv(csv_path)

	# Proxy de dopamina (consumo de alivio imediato)
	dopamine_cols = ["Eating_Out", "Entertainment", "Miscellaneous"]
	df["Gasto_Dopamina"] = df[dopamine_cols].sum(axis=1)
	df["Imposto_do_Cansaco_%"] = (df["Gasto_Dopamina"] / df["Income"]) * 100

	media_gasto = df["Gasto_Dopamina"].mean()
	media_imposto = df["Imposto_do_Cansaco_%"].mean()

	city_profile = (
		df.groupby("City_Tier", as_index=False)
		.agg(
			gasto_dopamina_medio=("Gasto_Dopamina", "mean"),
			imposto_cansaco_medio=("Imposto_do_Cansaco_%", "mean"),
			renda_media=("Income", "mean"),
			idade_media=("Age", "mean"),
			loan_medio=("Loan_Repayment", "mean"),
			n=("City_Tier", "size"),
		)
		.sort_values("City_Tier")
	)

	city_profile["desvio_gasto_%"] = (city_profile["gasto_dopamina_medio"] / media_gasto - 1) * 100
	city_profile["desvio_imposto_%"] = (city_profile["imposto_cansaco_medio"] / media_imposto - 1) * 100

	# Ponte com analise de idade: ponto aproximado de maturidade financeira
	idade_maturidade = 32
	df["Faixa_Maturidade"] = np.where(df["Age"] < idade_maturidade, "Antes dos 32", "32+")

	city_age_profile = (
		df.groupby(["City_Tier", "Faixa_Maturidade"], as_index=False)
		.agg(
			gasto_dopamina_medio=("Gasto_Dopamina", "mean"),
			imposto_cansaco_medio=("Imposto_do_Cansaco_%", "mean"),
			renda_media=("Income", "mean"),
			n=("Income", "size"),
		)
	)

	# Ponte com analise de ocupacao: mistura ocupacional por tier
	city_occ_profile = (
		df.groupby(["City_Tier", "Occupation"], as_index=False)
		.agg(
			imposto_cansaco_medio=("Imposto_do_Cansaco_%", "mean"),
			gasto_dopamina_medio=("Gasto_Dopamina", "mean"),
			n=("Occupation", "size"),
		)
	)

	tier_order = ["Tier_1", "Tier_2", "Tier_3"]
	city_profile = city_profile.set_index("City_Tier").reindex(tier_order).reset_index()
	city_age_profile["City_Tier"] = pd.Categorical(city_age_profile["City_Tier"], categories=tier_order, ordered=True)
	city_age_profile = city_age_profile.sort_values(["City_Tier", "Faixa_Maturidade"])

	fig = make_subplots(
		rows=1,
		cols=2,
		subplot_titles=(
			"Desvio relativo do gasto dopamina por cidade",
			"Desvio relativo do imposto do cansaco por cidade",
		),
	)

	fig.add_trace(
		go.Bar(
			x=city_profile["desvio_gasto_%"],
			y=city_profile["City_Tier"],
			orientation="h",
			marker_color=["#D64541" if x < 0 else "#2D6CDF" for x in city_profile["desvio_gasto_%"]],
			text=city_profile["desvio_gasto_%"].round(2).astype(str) + "%",
			textposition="outside",
			name="Desvio Gasto",
		),
		row=1,
		col=1,
	)

	fig.add_trace(
		go.Bar(
			x=city_profile["desvio_imposto_%"],
			y=city_profile["City_Tier"],
			orientation="h",
			marker_color=["#D64541" if x < 0 else "#F28E2B" for x in city_profile["desvio_imposto_%"]],
			text=city_profile["desvio_imposto_%"].round(2).astype(str) + "%",
			textposition="outside",
			name="Desvio Imposto",
		),
		row=1,
		col=2,
	)

	fig.update_layout(
		title="Dopamina e localizacao: comparacao relativa por City_Tier",
		template="plotly_white",
		height=580,
		showlegend=False,
	)
	fig.update_xaxes(title_text="Desvio percentual vs media geral", zeroline=True, zerolinecolor="gray", row=1, col=1)
	fig.update_xaxes(title_text="Desvio percentual vs media geral", zeroline=True, zerolinecolor="gray", row=1, col=2)
	fig.update_yaxes(title_text="City_Tier", row=1, col=1)
	fig.update_yaxes(title_text="City_Tier", row=1, col=2)
	fig.show()

	fig2 = go.Figure()
	for faixa in ["Antes dos 32", "32+"]:
		subset = city_age_profile[city_age_profile["Faixa_Maturidade"] == faixa]
		fig2.add_trace(
			go.Scatter(
				x=subset["City_Tier"],
				y=subset["imposto_cansaco_medio"],
				mode="lines+markers",
				name=faixa,
			)
		)

	fig2.update_layout(
		title="Imposto do cansaco por cidade: antes vs depois da maturidade financeira (32 anos)",
		template="plotly_white",
		xaxis_title="City_Tier",
		yaxis_title="Imposto do Cansaco (%)",
	)
	fig2.show()

	print("Resumo por localizacao (City_Tier):")
	print(
		city_profile[
			[
				"City_Tier",
				"gasto_dopamina_medio",
				"desvio_gasto_%",
				"imposto_cansaco_medio",
				"desvio_imposto_%",
				"renda_media",
				"idade_media",
				"loan_medio",
				"n",
			]
		].round(2).to_string(index=False)
	)

	print("\nConexao com analise de idade (ponto de maturidade ~32):")
	print(city_age_profile.round(2).to_string(index=False))

	print("\nConexao com analise de ocupacao (imposto medio por cidade e ocupacao):")
	print(city_occ_profile.pivot(index="Occupation", columns="City_Tier", values="imposto_cansaco_medio").round(2).to_string())

	top_city = city_profile.sort_values("desvio_gasto_%", ascending=False).iloc[0]
	low_city = city_profile.sort_values("desvio_gasto_%", ascending=True).iloc[0]

	print("\nLeitura executiva integrada:")
	print(f"- Cidade com maior desvio de gasto dopamina: {top_city['City_Tier']} ({top_city['desvio_gasto_%']:.2f}% vs media).")
	print(f"- Cidade com menor desvio de gasto dopamina: {low_city['City_Tier']} ({low_city['desvio_gasto_%']:.2f}% vs media).")
	print("- Assim como na analise por ocupacao, as diferencas percentuais sao moderadas, indicando efeito estrutural do consumo de alivio imediato.")
	print("- Em linha com a analise por idade, a localizacao altera a intensidade do imposto do cansaco, mas nao muda o padrao geral de persistencia desse gasto ao longo do ciclo de vida.")


if __name__ == "__main__":
	main()
